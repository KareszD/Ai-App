// App.js

import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  View,
  Alert,
  FlatList,Platform ,
} from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import axios from 'axios';
import * as Progress from 'react-native-progress';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import {
  Provider as PaperProvider,
  Appbar,
  TextInput,
  Button,
  Text,
  List,
  Avatar,
  ActivityIndicator,
  Snackbar,
  Dialog,
  Portal,
} from 'react-native-paper';
import * as MediaLibrary from 'expo-media-library';
import * as Sharing from 'expo-sharing';


const App = () => {
  // State variables
  const [apiIP, setApiIP] = useState('');
  const [apiIPSaved, setApiIPSaved] = useState(false);
  const [apiStatus, setApiStatus] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [message, setMessage] = useState('');
  const [progress, setProgress] = useState(0);
  const [estimatedTime, setEstimatedTime] = useState('');
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState([]);
  const [startTime, setStartTime] = useState(null);
  const [snackbarVisible, setSnackbarVisible] = useState(false);
  const [ipDialogVisible, setIpDialogVisible] = useState(false);

  // Load saved API IP from AsyncStorage
  useEffect(() => {
    const loadApiIP = async () => {
      try {
        const savedIP = await AsyncStorage.getItem('apiIP');
        if (savedIP) {
          setApiIP(savedIP);
          setApiIPSaved(true);
        } else {
          setIpDialogVisible(true);
        }
      } catch (error) {
        console.error('Failed to load API IP from storage', error);
        setIpDialogVisible(true);
      }
    };
    loadApiIP();
  }, []);

  // Function to handle API IP address submission
  const handleApiIPSubmit = async () => {
    if (apiIP) {
      setApiIPSaved(true);
      try {
        await AsyncStorage.setItem('apiIP', apiIP);
        setIpDialogVisible(false);
        checkApiStatus();
      } catch (error) {
        console.error('Failed to save API IP to storage', error);
      }
    } else {
      Alert.alert('Please enter a valid API IP address.');
    }
  };

  // Function to change the IP address
  const handleChangeIP = async () => {
    try {
      await AsyncStorage.removeItem('apiIP');
      setApiIPSaved(false);
      setApiIP('');
      setIpDialogVisible(true);
    } catch (error) {
      console.error('Failed to remove API IP from storage', error);
    }
  };

  // Function to check API status
  const checkApiStatus = () => {
    const apiUrl = `http://${apiIP}:5000/status`;

    axios
      .get(apiUrl)
      .then(response => {
        if (response.status === 200) {
          setApiStatus(true);
        } else {
          setApiStatus(false);
        }
      })
      .catch(error => {
        console.error('API Status Check Error:', error);
        setApiStatus(false);
      });
  };

  // Periodically check API status
  useEffect(() => {
    if (apiIPSaved) {
      // Check API status immediately
      checkApiStatus();

      // Set interval to check API status every 5 seconds
      const intervalId = setInterval(checkApiStatus, 5000);

      // Clean up the interval on unmount or when apiIP changes
      return () => clearInterval(intervalId);
    }
  }, [apiIPSaved, apiIP]);

  // Function to select files
  const selectFiles = async () => {
    try {
      const res = await DocumentPicker.getDocumentAsync({
        type: '*/*', // Allow any file type
        multiple: true, // Multiple selection is supported
      });

      console.log('Document Picker Response:', res);

      if (!res.canceled) {
        if (res.assets && res.assets.length > 0) {
          // Files were selected
          setSelectedFiles(res.assets);
          setMessage(`Selected ${res.assets.length} file(s)`);
          setSnackbarVisible(true);
        } else {
          setMessage('No files were selected.');
          setSnackbarVisible(true);
        }
      } else {
        setMessage('File selection was cancelled.');
        setSnackbarVisible(true);
      }
    } catch (err) {
      console.error('File Picker Error:', err);
      setMessage('Error selecting files.');
      setSnackbarVisible(true);
    }
  };

  // Function to send files to API
  const sendFiles = () => {
    console.log(selectedFiles.length)
    if (selectedFiles.length > 0) {
      if (processing) {
        setMessage('Processing is already ongoing. Please wait.');
        setSnackbarVisible(true);
        return;
      }

      setProcessing(true);
      setProgress(0);
      setEstimatedTime('');
      setResults([]);
      setStartTime(Date.now());

      // Start processing each file sequentially
      processNextFile(0);
    } else {
      setMessage('Please select files first.');
      setSnackbarVisible(true);
    }
  };

  // Function to process files sequentially
  const processNextFile = index => {
    if (index >= selectedFiles.length) {
      setProcessing(false);
      setMessage('All files have been processed.');
      setSnackbarVisible(true);
      return;
    }

    const file = selectedFiles[index];
    uploadFile(file)
      .then(() => pollProgress(index))
      .catch(error => {
        setMessage(`Error processing file: ${error}`);
        setProcessing(false);
        setSnackbarVisible(true);
      });
  };

  // Function to upload a single file
// Function to upload a single file
function uploadFile(file) {
  return new Promise(async (resolve, reject) => {
    // Replace 'localhost' with your actual server IP
    const apiUrl = `http://${apiIP}:5000/upload`;
    console.log('API URL:', apiUrl);

    const fileUri = file.uri;
    const fileName = file.name || fileUri.split('/').pop();
    const mimeType = file.mimeType || 'application/octet-stream';

    // Handle Android URI if necessary
    let uri = fileUri;
    if (Platform.OS === 'android') {
      uri = await FileSystem.getContentUriAsync(fileUri);
    }

    const formData = new FormData();

    formData.append('image', {
      uri: uri,
      name: fileName,
      type: mimeType,
    });

    console.log('FormData:', formData);

    fetch(apiUrl, {
      method: 'POST',
      body: formData,
      // Do not set 'Content-Type' header when using FormData with fetch
    })
      .then(response => response.json())
      .then(data => {
        if (data.message === 'Image uploaded successfully') {
          resolve();
        } else {
          reject(data.error || 'Unknown error');
        }
      })
      .catch(error => {
        console.error('Upload Error:', error);
        reject(error);
      });
  });
}


  

  // Function to poll progress
  const pollProgress = index => {
    let intervalId = setInterval(() => {
      const apiUrl = `http://${apiIP}:5000/all_progress`;

      axios
        .get(apiUrl)
        .then(response => {
          const data = response.data;
          if (data.progress === -1) {
            clearInterval(intervalId);
            setMessage(`Error processing file: ${data.result}`);
            setProcessing(false);
            setSnackbarVisible(true);
            return;
          }

          if (data.progress !== undefined) {
            setProgress(data.progress / 100);
          }

          // Update estimated time
          if (data.progress > 0 && data.progress < 100) {
            const elapsed = (Date.now() - startTime) / 1000; // in seconds
            const estimatedTotal = (elapsed / data.progress) * 100;
            const remaining = estimatedTotal - elapsed;

            const minutes = Math.floor(remaining / 60);
            const seconds = Math.round(remaining % 60);

            setEstimatedTime(
              `Estimated time remaining: ${minutes} minute(s) and ${seconds} second(s)`
            );
          } else if (data.progress === 100) {
            clearInterval(intervalId);
            fetchResult()
              .then(() => {
                // Proceed to next file
                processNextFile(index + 1);
              })
              .catch(error => {
                setMessage(`Error fetching result: ${error}`);
                setProcessing(false);
                setSnackbarVisible(true);
              });
          }
        })
        .catch(error => {
          clearInterval(intervalId);
          setMessage('Error fetching progress.');
          console.error('Progress Polling Error:', error);
          setProcessing(false);
          setSnackbarVisible(true);
        });
    }, 1000); // Poll every second
  };

  // Function to fetch result after completion
  const fetchResult = () => {
    const apiUrl = `http://${apiIP}:5000/result`;

    return axios
      .get(apiUrl)
      .then(response => {
        const data = response.data;
        if (data.filename) {
          displayResult(data.filename);
        } else {
          displayResult(null, data.result);
        }
      })
      .catch(error => {
        console.error('Result Fetch Error:', error);
        throw error;
      });
  };

  // Function to display the result
  const displayResult = (filename, messageText = 'Prediction completed successfully.') => {
    if (filename) {
      const resultZipPath = `http://${apiIP}:5000/results/output.zip`;

      setResults(prevResults => [
        ...prevResults,
        {
          message: messageText,
          filename: filename,
          resultZipPath: resultZipPath,
        },
      ]);
    } else {
      setResults(prevResults => [
        ...prevResults,
        {
          message: messageText,
        },
      ]);
    }
  };

  const downloadFile = async (url, filename) => {
    try {
      // Download the file to a temporary location
      const fileUri = FileSystem.cacheDirectory + filename;
      const downloadResumable = FileSystem.createDownloadResumable(url, fileUri);
  
      const { uri } = await downloadResumable.downloadAsync();
      if (!uri) {
        throw new Error('File download failed');
      }
  
      // Share the file
      const available = await Sharing.isAvailableAsync();
      if (available) {
        await Sharing.shareAsync(uri);
      } else {
        Alert.alert('Sharing not available', 'Your device does not support sharing files.');
      }
    } catch (error) {
      console.error('Download Error:', error);
      Alert.alert('Download Failed', error.message || 'An error occurred while downloading the file.');
    }
  };

  // Function to dismiss snackbar
  const dismissSnackbar = () => {
    setSnackbarVisible(false);
  };

  return (
    <PaperProvider>
      <SafeAreaProvider>
        <SafeAreaView style={styles.safeArea}>
          <Appbar.Header>
            <Appbar.Content title="Image Upload App" />
            {apiIPSaved && (
              <>
                <Avatar.Icon
                  size={24}
                  icon="circle"
                  color={apiStatus ? 'green' : 'red'}
                  style={{ backgroundColor: 'transparent' }}
                />
                <Appbar.Action icon="pencil" onPress={handleChangeIP} />
              </>
            )}
          </Appbar.Header>

          {!apiIPSaved ? (
            <Portal>
              <Dialog visible={!apiIPSaved} dismissable={false}>
                <Dialog.Title>Enter API Server IP Address</Dialog.Title>
                <Dialog.Content>
                  <TextInput
                    label="API Server IP"
                    value={apiIP}
                    onChangeText={setApiIP}
                    keyboardType="numeric"
                  />
                </Dialog.Content>
                <Dialog.Actions>
                  <Button onPress={handleApiIPSubmit}>Submit</Button>
                </Dialog.Actions>
              </Dialog>
            </Portal>
          ) : (
            <View style={styles.container}>
              <Button
                mode="contained"
                onPress={selectFiles}
                style={styles.button}
                icon="file-upload"
              >
                Select Files
              </Button>

              {selectedFiles.length > 0 && (
                <List.Section>
                  <List.Subheader>Selected Files</List.Subheader>
                  {selectedFiles.map((item, index) => (
                    <List.Item
                      key={index}
                      title={item.name}
                      left={() => <List.Icon icon="file" />}
                    />
                  ))}
                </List.Section>
              )}

              <Button
                mode="contained"
                onPress={sendFiles}
                style={styles.button}
                icon="cloud-upload"
                disabled={!apiStatus || processing}
              >
                Send Files to API
              </Button>

              {processing && (
                <View style={styles.progressContainer}>
                  <Progress.Bar progress={progress} width={200} />
                  <Text>{estimatedTime}</Text>
                  <ActivityIndicator animating={true} style={styles.activityIndicator} />
                </View>
              )}

              {results.length > 0 && (
                <List.Section>
                  <List.Subheader>Results</List.Subheader>
                  {results.map((result, index) => (
                    <List.Item
                      key={index}
                      title={result.message}
                      right={() =>
                        result.filename ? (
                          <Button
                            mode="text"
                            onPress={() => downloadFile(result.resultZipPath, result.filename)}
                          >
                            Download
                          </Button>
                        ) : null
                      }
                    />
                  ))}
                </List.Section>
              )}
            </View>
          )}

          <Snackbar
            visible={snackbarVisible}
            onDismiss={dismissSnackbar}
            duration={3000}
          >
            {message}
          </Snackbar>
        </SafeAreaView>
      </SafeAreaProvider>
    </PaperProvider>
  );
};

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
  },
  container: {
    flex: 1,
    padding: 16,
  },
  button: {
    marginVertical: 10,
  },
  progressContainer: {
    alignItems: 'center',
    marginVertical: 20,
  },
  activityIndicator: {
    marginTop: 10,
  },
});

export default App;
