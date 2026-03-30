import React, { useState, useEffect } from 'react';
import { StatusBar, StyleSheet, ActivityIndicator, View, Text } from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import { WebView } from 'react-native-webview';
import * as Location from 'expo-location';

export default function App() {
  const PRODUCTION_URL = 'https://chikitsalya-frontend.onrender.com';
  const [injectedCode, setInjectedCode] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  useEffect(() => {
    (async () => {
      // 1. Request Permission
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setInjectedCode('true;'); // Permission denied, proceed anyway
        return;
      }

      try {
        // 2. Fetch Location with a 6-second timeout to prevent hanging
        const locationPromise = Location.getCurrentPositionAsync({ 
          accuracy: Location.Accuracy.Balanced 
        });
        
        const timeoutPromise = new Promise((_, reject) => 
          setTimeout(() => reject(new Error('GPS Timeout')), 6000)
        );

        // Race the GPS against the clock
        const loc = await Promise.race([locationPromise, timeoutPromise])
          .catch((err) => {
            console.log("GPS search timed out or failed, loading site without pre-fill.");
            return null; 
          });

        if (loc) {
          const fakeGeolocation = `
            (function() {
              const locData = {
                coords: {
                  latitude: ${loc.coords.latitude},
                  longitude: ${loc.coords.longitude},
                  accuracy: ${loc.coords.accuracy || 10}
                },
                timestamp: Date.now()
              };
              window.navigator.geolocation.getCurrentPosition = function(s, e) {
                setTimeout(() => s(locData), 0);
              };
              window.navigator.geolocation.watchPosition = function(s) {
                setTimeout(() => s(locData), 0);
                return 1;
              };
            })();
            true;
          `;
          setInjectedCode(fakeGeolocation);
        } else {
          setInjectedCode('true;'); // Timeout reached, load site without fake GPS
        }
      } catch (error) {
        setInjectedCode('true;'); // General error, proceed anyway
      }
    })();
  }, []);

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" />
        {injectedCode ? (
          <WebView
            source={{ uri: PRODUCTION_URL }}
            style={styles.webview}
            javaScriptEnabled={true}
            domStorageEnabled={true}
            geolocationEnabled={true}
            injectedJavaScriptBeforeContentLoaded={injectedCode}
            startInLoadingState={true}
            renderLoading={() => (
              <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color="#0ea5e9" />
                <Text style={styles.loadingText}>Connecting to Chikitsalya Cloud...</Text>
              </View>
            )}
          />
        ) : (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#0ea5e9" />
            <Text style={styles.loadingText}>Optimizing GPS Signal...</Text>
          </View>
        )}
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0B1120',
  },
  webview: {
    flex: 1,
  },
  loadingContainer: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#0B1120',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1,
  },
  loadingText: {
    color: '#94a3b8',
    marginTop: 15,
    fontSize: 14,
    fontWeight: '500',
  }
});
