// main.js
const { app, BrowserWindow } = require('electron');
const path = require('path');
const { PythonShell } = require('python-shell');
const { autoUpdater } = require('electron-updater');
const log = require('electron-log');

// Configure logging for autoUpdater
autoUpdater.logger = log;
autoUpdater.logger.transports.file.level = 'info';
log.info('App starting...');

let mainWindow;
let apiShell = null; // This will hold our PythonShell instance

// Function to create the main application window
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1000,
    height: 700,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false, // In production, consider enabling context isolation for added security
    },
  });

  mainWindow.loadFile('index.html');

  // Optionally, open DevTools:
  // mainWindow.webContents.openDevTools();

  // Check for updates when the window is ready
  mainWindow.once('ready-to-show', () => {
    autoUpdater.checkForUpdatesAndNotify();
  });
}

// Function to start the Python API using PythonShell
function startApi() {
  const pythonExecutable = 'python'; // Change to 'python3' if needed
  const scriptName = 'api.py'; // Your Python script in the "Py" folder

  // Options: Use unbuffered mode (-u) to immediately flush output
  const options = {
    pythonPath: pythonExecutable,
    scriptPath: path.join(__dirname, 'Py'),
    pythonOptions: ['-u'], // Unbuffered output for real-time logging
    // args: [] // Add any additional arguments here if needed
  };

  log.info(`Starting API using PythonShell with script: ${path.join(__dirname, 'Py', scriptName)}`);

  // Start the Python script
  apiShell = new PythonShell(scriptName, options);

  // Listen for messages from the Python process (for example, print() statements)
  apiShell.on('message', (message) => {
    log.info(`API message: ${message}`);
  });

  // Listen for error events
  apiShell.on('error', (err) => {
    log.error(`PythonShell error: ${err}`);
  });

  // When the Python process ends, log that fact
  apiShell.on('close', () => {
    log.info('PythonShell process closed.');
    apiShell = null;
  });
}

// Function to stop the Python API gracefully
function stopApi() {
  return new Promise((resolve) => {
    if (apiShell) {
      log.info('Stopping PythonShell API process...');
      apiShell.end((err, code, signal) => {
        if (err) {
          log.error(`Error terminating API process: ${err}`);
        } else {
          log.info(`API process terminated (code: ${code}, signal: ${signal}).`);
        }
        apiShell = null;
        resolve();
      });
    } else {
      resolve();
    }
  });
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
  if (mainWindow && mainWindow.webContents) {
    mainWindow.webContents.send('updateReady');
  }
  autoUpdater.quitAndInstall();
});

// Application lifecycle events
app.whenReady().then(() => {
  createWindow();
  startApi();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('before-quit', async (event) => {
  event.preventDefault(); // Prevent immediate quit so we can shut down the API gracefully.
  await stopApi();
  app.exit();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    stopApi().then(() => {
      app.quit();
    });
  }
});

// Handle termination signals for graceful shutdown
const gracefulShutdown = () => {
  log.info('Received termination signal.');
  stopApi().then(() => {
    app.quit();
  });
};

process.on('SIGINT', gracefulShutdown);
process.on('SIGTERM', gracefulShutdown);

app.on('quit', () => {
  stopApi();
});
