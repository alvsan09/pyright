import * as child_process from 'child_process';
import path from 'path';
import * as readline from 'readline';

import { ConfigOptions } from '../common/configOptions';
import { ConsoleInterface, StandardConsole } from '../common/console';
import { convertOffsetToPosition, convertPositionToOffset } from '../common/positionUtils';
import { Position } from '../common/textRange';
import { ParseResults } from '../parser/parser';

export enum AiModel {
	ngram = 'ngram',
	gpt2 = 'gpt2'
}

// We should find a way to automatically copy python scripts and model data in the dist directory

const srcPath = ['..', '..', 'pyright-internal', 'src', 'completion'];
const dataPath = ['..', '..', 'pyright-internal', 'data'];

const ngramScriptPath = [...srcPath, 'ngram-predict.py'];
const ngramModelPath = [...dataPath, 'data_train_1.0.2_n4.pkl'];

const gpt2ScriptPath = [...srcPath, 'gpt2-predict.py'];
const gpt2ModelPath = [...dataPath, 'gpt2'];

export abstract class AiCompleter {

	#pyScriptPath: string;
	#modelPath: string;
    #console: ConsoleInterface;
	#childProcess!: child_process.ChildProcess;
	#stdout!: readline.Interface;
	#callCount: number;
    #isReady: boolean;

	protected predictions: Map<string, string[]>;
	protected pendingPredictions: Map<string, Promise<string[]>>;

	protected constructor(pyScriptPath: string, modelPath: string, console: ConsoleInterface) {
		this.#pyScriptPath = pyScriptPath;
		this.#modelPath = modelPath;
        this.#console = console || new StandardConsole();
		this.#callCount = 0;
        this.#isReady = false;
		this.predictions = new Map();
		this.pendingPredictions = new Map();
	}

	static create(model: AiModel, configOptions: ConfigOptions, console: ConsoleInterface): AiCompleter {
		if (model === AiModel.ngram) {
			return new NgramCompleter(
				path.resolve(configOptions.projectRoot, ...ngramScriptPath),
				path.resolve(configOptions.projectRoot, ...ngramModelPath),
				console
			)
		} else if (model === AiModel.gpt2) {
			return new GPT2Completer(
				path.resolve(configOptions.projectRoot, ...gpt2ScriptPath),
				path.resolve(configOptions.projectRoot, ...gpt2ModelPath),
				console
			)
		} else {
			throw new TypeError(`${model} is not a valid AI model`);
		}
	}

	async start(): Promise<void> {
		this.#childProcess = child_process.spawn('python', [this.#pyScriptPath, this.#modelPath]);
		readline.createInterface(this.#childProcess.stderr!).on('line', message => this.#console.error(message));
		this.#stdout = readline.createInterface(this.#childProcess.stdout!);
		// wait for ready state event
		await new Promise<void>(resolve => {
			const listener = (line: string) => {
				const { event, value } = JSON.parse(line);
				if (event === 'state' && value === 'ready') {
					this.#stdout.off('line', listener);
                    this.#isReady = true;
					resolve();
				}
			};
			this.#stdout.on('line', listener);
		});
	}
	
	async predict(parseResults: ParseResults | undefined, position?: Position, requestResults = true): Promise<string[]> {
		if (!this.#isReady || !parseResults?.text.length) {
			return [];
		}

		const lines = parseResults.tokenizerOutput.lines;
		position = position || convertOffsetToPosition(lines.end, lines);

		const lastWordTyped = this.getLastWordTyped(parseResults, position);
		const context = this.getContext(parseResults, position, lastWordTyped);

		this.#console.info('context : ' + context);
		if (!context) {
			return [];
		}

		// Use to pre-compute context
		if (!requestResults) {
			await this.prepare(context);
			return [];
		}

		// Do not request predictions if an identical request is already pending
		if (this.pendingPredictions.has(context)) {
			return this.pendingPredictions.get(context)!;
		}

		if (!this.predictions.has(context)) {
			const promise = this.ghettoRpc('predict', { context });
			this.pendingPredictions.set(context, promise);
			const predictions = await promise;
			this.predictions.set(context, predictions);
			this.pendingPredictions.delete(context);

			// Now that predictions are calculated, prepare whole context in advance
			this.predict(parseResults, undefined, false)
		}

		return this.predictions.get(context)!;
	}

    get isReady() { return this.#isReady; }

	protected async ghettoRpc(method: string, params: any): Promise<any> {
		const id = ++this.#callCount;
		this.#childProcess.stdin!.write(JSON.stringify({ method, params, id }) + '\n');

		return new Promise(resolve => {
			const callback = (line: string) => {
				const res = JSON.parse(line);
				if (res?.event === method && res?.id === id) {
					this.#stdout.off('line', callback);
					resolve(res.value);
				}
			}

			this.#stdout.on('line', callback);
		});
	}

	protected lineAt(parseResults: ParseResults, position: Position): string {
		const line = parseResults.tokenizerOutput.lines.getItemAt(position.line);
		return parseResults.text.substring(line.start, line.start + line.length);
	}

	protected getLastWordTyped(parseResults: ParseResults, position: Position): string {
		const match = this.lineAt(parseResults, position).substring(0, position.character).match(/\w+$/);
		return match ? match[0] : '';
	}
	
	protected async prepare(context: string): Promise<void> { 
		// Do nothing
	}

	protected abstract getContext(parseResults: ParseResults, position: Position, lastWordTyped?: string): string;
}

class NgramCompleter extends AiCompleter {
	protected getContext(parseResults: ParseResults, position: Position, lastWordTyped?: string): string {
		return this.lineAt(parseResults, position).substring(0, position.character - (lastWordTyped?.length || 0)).trim();
	}
}

class GPT2Completer extends AiCompleter {
	protected override async prepare(context: string): Promise<void> {
		if (!this.pendingPredictions.has(context) || !this.predictions.has(context)) {
			await this.ghettoRpc('prepare', { context });
		}
	}

	protected getContext(parseResults: ParseResults, position: Position, lastWordTyped?: string): string {
		position.character -= lastWordTyped?.length || 0;
		// Trim trailing whitespace or endline
		return parseResults.text.substring(0, convertPositionToOffset(position, parseResults.tokenizerOutput.lines)).trim();
	}
}