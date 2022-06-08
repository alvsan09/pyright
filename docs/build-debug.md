## Building Pyright

To install the dependencies for all packages in the repo:
1. Install [nodejs](https://nodejs.org/en/) version 14.x
2. Open terminal window in main directory of cloned source
3. Execute `npm install` to install dependencies

### Requirements for Language Model

I haven't taken the time to package it properly inside Pyright so you have to install Python and the necessary dependencies by yourself
1. Install Python 3.8
2. Install the following libraries using pip:
    - `dill`
    - `nltk`
    - `transformers`
3. **ngram** : Download the `pkl` file and put it under `packages/pyright-internal/data/data_train_1.0.2_n4.pkl` <br>
   **gpt2** : Download `gpt2_finetune.tar.gz` and extract it under `packages/pyright-internal/data/gpt2`


## Building the CLI

1. cd to the `packages/pyright` directory
2. Execute `npm run build`

Once built, you can run the command-line tool by executing the following:

`node index.js`

## Building the VS Code extension

1. cd to the `packages/vscode-pyright` directory
2. Execute `npm run package`

The resulting package (pyright-X.Y.Z.vsix) can be found in the client directory.
To install in VS Code, go to the extensions panel and choose “Install from VSIX...” from the menu, then select the package.


## Running Pyright tests

1. cd to the `packages/pyright-internal` directory
2. Execute `npm run test`


## Debugging Pyright

To debug pyright, open the root source directory within VS Code. Open the debug sub-panel and choose “Pyright CLI” from the debug target menu. Click on the green “run” icon or press F5 to build and launch the command-line version in the VS Code debugger.

To debug the VS Code extension, select “Pyright extension” from the debug target menu. Click on the green “run” icon or press F5 to build and launch a second copy of VS Code with the extension. Within the second VS Code instance, open a python source file so the pyright extension is loaded. Return to the first instance of VS Code and select “Pyright extension attach server” from the debug target menu and click the green “run” icon. This will attach the debugger to the process that hosts the type checker. You can now set breakpoints, etc.

To debug the VS Code extension in watch mode, you can do the above, but select “Pyright extension (watch mode)”. When pyright's source is saved, an incremental build will occur, and you can either reload the second VS Code window or relaunch it to start using the updated code. Note that the watcher stays open when debugging stops, so you may need to stop it (or close VS Code) if you want to perform packaging steps without the output potentially being overwritten.
