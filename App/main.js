// main.js
const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const log = require('electron-log'); // Optional for better logging
const { exec } = require('child_process');

let mainWindow;
let apiProcess = null;

// Function to create the main application window
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1000,
    height: 700,
    webPreferences: {
      //preload: path.join(__dirname, 'preload.js'), // If using preload scripts
      nodeIntegration: true, // Adjust based on your security needs
      contextIsolation: false, // Adjust based on your security needs
    },
  });

  mainWindow.loadFile('index.html');

  // Open DevTools (optional)
   mainWindow.webContents.openDevTools();
}

// Function to start the Flask API
function startApi() {
  const apiPath = path.join(__dirname, 'Py', 'start.bat');
  const options = {
    cwd: path.dirname(apiPath),
  };

  console.log(`Starting API using batch file at: ${apiPath}`);
  console.log(`Working directory set to: ${options.cwd}`);

  exec(`"${apiPath}"`, options, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error executing batch file: ${error.message}`);
      return;
    }

    if (stderr) {
      console.error(`Batch file stderr: ${stderr}`);
      // Continue processing; stderr may contain warnings
    }

    console.log(`Batch file output:\n${stdout}`);
  });
}


// Function to stop the Flask API
function stopApi() {
  if (apiProcess) {
    log.info('Stopping API subprocess...');
    console.log('Stopping API subprocess...');

    // Send termination signal
    apiProcess.kill('SIGINT');

    // Optionally, set a timeout to force kill if it doesn't terminate
    const killTimeout = setTimeout(() => {
      if (apiProcess) {
        log.warn('Force killing API subprocess...');
        console.warn('Force killing API subprocess...');
        apiProcess.kill('SIGKILL');
      }
    }, 5000); // 5 seconds

    // Clear timeout if process exits gracefully
    apiProcess.on('exit', () => {
      clearTimeout(killTimeout);
      log.info('API subprocess terminated.');
      console.log('API subprocess terminated.');
      apiProcess = null;
    });
  }
}

// Function to get the Python executable path
function getPythonExecutable() {
  const platform = process.platform;

  if (platform === 'win32') {
    return 'python'; // Assumes python is in PATH
    // Alternatively, specify the full path: 'C:\\Path\\To\\python.exe'
  } else {
    return 'python3'; // Common on Unix-like systems
    // Alternatively, specify the full path: '/usr/bin/python3'
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
  if (process.platform !== 'darwin') app.quit();
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
