const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const kill = require('tree-kill');
const { autoUpdater } = require('electron-updater');
const log = require('electron-log');

// Configure logging for autoUpdater
autoUpdater.logger = log;
autoUpdater.logger.transports.file.level = 'info';
log.info('App starting...');

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
  //mainWindow.webContents.openDevTools();

  // Check for updates when the window is ready
  mainWindow.once('ready-to-show', () => {
    autoUpdater.checkForUpdatesAndNotify();
  });
}

// Function to start the Flask API
function startApi() {
  const pythonExecutable = 'python';
  const apiPath = path.join(__dirname, 'Py', 'api.py');
  const options = {
    cwd: path.dirname(apiPath),
    shell: false,
  };

  console.log(`Starting API using script at: ${apiPath}`);
  console.log(`Working directory set to: ${options.cwd}`);
  if (startapibool) {
    apiProcess = spawn(pythonExecutable, [apiPath], options);
    apiProcess.unref();
    console.log(`API process started with PID: ${apiProcess.pid}`);

    apiProcess.stdout.on('data', (data) => {
      console.log(`API stdout: ${data}`);
    });

    apiProcess.stderr.on('data', (data) => {
      console.error(`API stderr: ${data}`);
    });

    apiProcess.on('close', (code) => {
      console.log(`API process exited with code ${code}`);
      apiProcess = null;
    });

    apiProcess.on('error', (err) => {
      console.error(`Failed to start API process: ${err}`);
      apiProcess = null;
    });
  }
}

// Function to stop the Flask API
function stopApi() {
  if (apiProcess) {
    console.log(`Stopping API process with PID: ${apiProcess.pid}`);
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

// Auto-update event listeners
autoUpdater.on('checking-for-update', () => {
  log.info('Checking for updates...');
});

autoUpdater.on('update-available', (info) => {
  log.info('Update available:', info);
});

autoUpdater.on('update-not-available', (info) => {
  log.info('Update not available:', info);
});

autoUpdater.on('error', (err) => {
  log.error('Error in auto-updater:', err);
});

autoUpdater.on('download-progress', (progressObj) => {
  let logMessage = `Download speed: ${progressObj.bytesPerSecond} - Downloaded ${progressObj.percent}% (${progressObj.transferred}/${progressObj.total})`;
  log.info(logMessage);
});

autoUpdater.on('update-downloaded', (info) => {
  log.info('Update downloaded:', info);
  // Notify user and install update
  mainWindow.webContents.send('updateReady');
  autoUpdater.quitAndInstall();
});

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
  if (process.platform !== 'darwin') {
    stopApi();
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
