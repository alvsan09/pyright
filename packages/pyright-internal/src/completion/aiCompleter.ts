import * as child_process from 'child_process';
import * as readline from 'readline';

import { ConsoleInterface, StandardConsole } from '../common/console';
import { Position } from '../common/textRange';
import { ParseResults } from '../parser/parser';

export class AiCompleter {

	#pyScriptPath: string;
	#modelPath: string;
    #console: ConsoleInterface;
	#childProcess!: child_process.ChildProcess;
	#stdout!: readline.Interface;
    #isReady: boolean;

	constructor(pyScriptPath: string, modelPath: string, console: ConsoleInterface) {
		this.#pyScriptPath = pyScriptPath;
		this.#modelPath = modelPath;
        this.#console = console || new StandardConsole();
        this.#isReady = false;
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
	
	async predict(parseResults: ParseResults | undefined, position: Position): Promise<string[]> {
		if (!parseResults) {
			return [];
		}

		const match = this.#lineAt(parseResults, position).substring(0, position.character).match(/\w+$/);
		const lastWordTyped = match ? match[0] : '';

		const context = this.#lineAt(parseResults, position).substring(0, position.character - lastWordTyped.length).trim();

		this.#console.info('context : ' + context);
		if (!context) {
			return [];
		}
		return this.#ghettoRpc('predict', { context });
	}

    get isReady() { return this.#isReady; }

	async #ghettoRpc(method: string, params: any): Promise<any> {
		this.#childProcess.stdin!.write(JSON.stringify({ method, params }) + '\n');
		return new Promise(resolve => {
			this.#stdout.once('line', line => resolve(JSON.parse(line)));
		});
	}

	#lineAt(parseResults: ParseResults, position: Position): string {
		const line = parseResults.tokenizerOutput.lines.getItemAt(position.line);
		return parseResults.text.substring(line.start, line.start + line.length);
	}
}