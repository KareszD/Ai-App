const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const kill = require('tree-kill');

let mainWindow;
let apiProcess = null;
let startapibool = false;
// Function to create the main application window
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1000,
    height: 700,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  mainWindow.loadFile('index.html');

  // Optionally, open DevTools
   mainWindow.webContents.openDevTools();
}

// Function to start the Flask API
function startApi() {
  const apiPath = path.join(__dirname, 'Py', 'start.bat');
  const options = {
    cwd: path.dirname(apiPath),
    shell: true,
  };

  console.log(`Starting API using batch file at: ${apiPath}`);
  console.log(`Working directory set to: ${options.cwd}`);
  if(startapibool){
  // Spawn the batch file process
  apiProcess = spawn(apiPath, [], options);

  console.log(`API process started with PID: ${apiProcess.pid}`);

  apiProcess.stdout.on('data', (data) => {
    console.log(`API stdout: ${data}`);
  });

  apiProcess.stderr.on('data', (data) => {
    console.error(`API stderr: ${data}`);
  });

  apiProcess.on('close', (code) => {
    console.log(`API process exited with code ${code}`);
    apiProcess = null; // Reset the process variable
  });

  apiProcess.on('error', (err) => {
    console.error(`Failed to start API process: ${err}`);
    apiProcess = null;
  });}
}

// Function to stop the Flask API
function stopApi() {
  if (apiProcess) {
    console.log(`Stopping API process with PID: ${apiProcess.pid}`);

    // Use tree-kill to terminate the process and its child processes
    kill(apiProcess.pid, 'SIGTERM', (err) => {
      if (err) {
        console.error(`Failed to terminate process: ${err.message}`);
      } else {
        console.log(`API process terminated.`);
      }
      apiProcess = null;
    });
  }
}

// Handle application ready event
app.whenReady().then(() => {
  createWindow();
  startApi();

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

// Handle application before-quit event to stop the API
app.on('before-quit', () => {
  stopApi();
});

// Handle all windows closed event
app.on('window-all-closed', function () {
  // On macOS, it's common for applications to stay open until the user quits explicitly
  if (process.platform !== 'darwin') {
    stopApi(); // Stop the API process before quitting
    app.quit();
  }
});

// Handle termination signals
process.on('SIGINT', () => {
  console.log('Received SIGINT.');
  stopApi();
  app.quit();
});

process.on('SIGTERM', () => {
  console.log('Received SIGTERM.');
  stopApi();
  app.quit();
});

app.on('quit', () => {
  stopApi();
});
