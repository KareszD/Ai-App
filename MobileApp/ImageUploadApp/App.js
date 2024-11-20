// App.js

import React, { useState, useEffect } from 'react';
import {
  SafeAreaView,
  StyleSheet,
  View,
  Text,
  Button,
  TextInput,
  Image,
  TouchableOpacity,
  ScrollView,
  FlatList,
  Alert,
} from 'react-native';
import * as ImagePicker from 'react-native-image-picker';
import axios from 'axios';
import * as Progress from 'react-native-progress';
import AsyncStorage from '@react-native-async-storage/async-storage';

const App = () => {
  const [apiIP, setApiIP] = useState('');
  const [apiIPSaved, setApiIPSaved] = useState(false);
  const [selectedImages, setSelectedImages] = useState([]);
  const [message, setMessage] = useState('Hello World!');
  const [progress, setProgress] = useState(0);
  const [estimatedTime, setEstimatedTime] = useState('');
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState([]);
  const [startTime, setStartTime] = useState(null);

  // Load saved API IP from AsyncStorage
  useEffect(() => {
    const loadApiIP = async () => {
      try {
        const savedIP = await AsyncStorage.getItem('apiIP');
        if (savedIP) {
          setApiIP(savedIP);
          setApiIPSaved(true);
        }
      } catch (error) {
        console.error('Failed to load API IP from storage', error);
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
      } catch (error) {
        console.error('Failed to save API IP to storage', error);
      }
    } else {
      Alert.alert('Please enter a valid API IP address.');
    }
  };

  // Function to select images
  const selectImages = () => {
    const options = {
      mediaType: 'photo',
      selectionLimit: 0, // Set to 0 to allow multiple images
    };

    ImagePicker.launchImageLibrary(options, response => {
      if (response.didCancel) {
        console.log('User cancelled image picker');
      } else if (response.errorCode) {
        console.log('ImagePicker Error: ', response.errorMessage);
      } else {
        if (response.assets) {
          setSelectedImages(response.assets);
          setMessage(`Selected ${response.assets.length} file(s)`);
        } else {
          setMessage('Please select images to upload.');
        }
      }
    });
  };

  // Function to send images to API
  const sendImages = () => {
    if (selectedImages.length > 0) {
      if (processing) {
        setMessage('Processing is already ongoing. Please wait.');
        return;
      }

      setProcessing(true);
      setProgress(0);
      setEstimatedTime(`Estimated time remaining: ${selectedImages.length * 0} seconds`);
      setResults([]);
      setStartTime(Date.now());

      // Start processing each image sequentially
      processNextImage(0);
    } else {
      setMessage('Please select images first.');
    }
  };

  // Function to process images sequentially
  const processNextImage = index => {
    if (index >= selectedImages.length) {
      setProcessing(false);
      setMessage('All images have been processed.');
      return;
    }

    const image = selectedImages[index];
    uploadImage(image)
      .then(() => pollProgress(index))
      .catch(error => {
        setMessage(`Error processing image ${image.fileName}: ${error}`);
        setProcessing(false);
      });
  };

  // Function to upload a single image
  const uploadImage = image => {
    return new Promise((resolve, reject) => {
      const apiUrl = `http://${apiIP}:5000/upload`;

      const formData = new FormData();
      formData.append('image', {
        uri: image.uri,
        type: image.type,
        name: image.fileName,
      });

      axios
        .post(apiUrl, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        })
        .then(response => {
          if (response.data.message === 'Image uploaded successfully') {
            resolve();
          } else {
            reject(response.data.error || 'Unknown error');
          }
        })
        .catch(error => {
          console.error('Upload Error:', error);
          reject(error);
        });
    });
  };

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
            setMessage(`Error processing image: ${data.result}`);
            setProcessing(false);
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

            setEstimatedTime(`Estimated time remaining: ${minutes} minute(s) and ${seconds} second(s)`);
          } else if (data.progress === 100) {
            clearInterval(intervalId);
            fetchResult()
              .then(() => {
                // Proceed to next image
                processNextImage(index + 1);
              })
              .catch(error => {
                setMessage(`Error fetching result: ${error}`);
                setProcessing(false);
              });
          }
        })
        .catch(error => {
          clearInterval(intervalId);
          setMessage('Error fetching progress.');
          console.error('Progress Polling Error:', error);
          setProcessing(false);
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

  // Function to check API status periodically
  const checkApiStatus = () => {
    const apiUrl = `http://${apiIP}:5000/status`;

    axios
      .get(apiUrl)
      .then(response => {
        if (response.status === 200) {
          // API is online
          // Do something if needed
        } else {
          // API is offline
          // Do something if needed
        }
      })
      .catch(error => {
        console.error('API Status Check Error:', error);
      });
  };

  useEffect(() => {
    if (apiIPSaved) {
      // Periodically check API status every 30 seconds
      const intervalId = setInterval(checkApiStatus, 30000); // 30 seconds

      // Clean up the interval on unmount
      return () => clearInterval(intervalId);
    }
  }, [apiIPSaved]);

  if (!apiIPSaved) {
    // Render IP address input screen
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.heading}>Enter API Server IP Address</Text>
        <TextInput
          style={styles.input}
          placeholder="API Server IP"
          value={apiIP}
          onChangeText={setApiIP}
        />
        <Button title="Submit" onPress={handleApiIPSubmit} />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.heading}>{message}</Text>
      <Button title="Select Images" onPress={selectImages} />

      {selectedImages.length > 0 && (
        <FlatList
          data={selectedImages}
          keyExtractor={item => item.uri}
          renderItem={({ item }) => (
            <Image source={{ uri: item.uri }} style={styles.imageThumbnail} />
          )}
          horizontal
        />
      )}

      <Button title="Send Images to API" onPress={sendImages} />

      {processing && (
        <View style={styles.progressContainer}>
          <Progress.Bar progress={progress} width={200} />
          <Text>{estimatedTime}</Text>
        </View>
      )}

      {results.length > 0 && (
        <ScrollView style={styles.resultContainer}>
          <Text style={styles.heading}>Prediction Results</Text>
          {results.map((result, index) => (
            <View key={index} style={styles.resultItem}>
              <Text>{result.message}</Text>
              {result.filename && (
                <TouchableOpacity
                  onPress={() => {
                    // Handle download action
                    Alert.alert('Download', 'Download link: ' + result.resultZipPath);
                  }}
                  style={styles.downloadButton}
                >
                  <Text style={styles.downloadButtonText}>Download Shapefile ZIP</Text>
                </TouchableOpacity>
              )}
            </View>
          ))}
        </ScrollView>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
  },
  heading: {
    fontSize: 20,
    marginVertical: 10,
    textAlign: 'center',
  },
  input: {
    height: 40,
    borderWidth: 1,
    paddingHorizontal: 8,
    marginVertical: 10,
  },
  imageThumbnail: {
    width: 100,
    height: 100,
    margin: 5,
  },
  progressContainer: {
    alignItems: 'center',
    marginVertical: 20,
  },
  resultContainer: {
    marginTop: 20,
  },
  resultItem: {
    marginBottom: 10,
  },
  downloadButton: {
    backgroundColor: 'blue',
    padding: 10,
    marginTop: 5,
  },
  downloadButtonText: {
    color: 'white',
    textAlign: 'center',
  },
});

export default App;
