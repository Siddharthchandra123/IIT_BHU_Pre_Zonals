import React, { useState, useEffect } from 'react';
import { StatusBar, StyleSheet, ActivityIndicator, View } from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import { WebView } from 'react-native-webview';
import * as Location from 'expo-location';

export default function App() {
  // PRODUCTION URL: Now pointing to your live Render.com Cloud Server!
  const PRODUCTION_URL = 'https://chikitsalya-frontend.onrender.com';
  const [injectedCode, setInjectedCode] = useState(null);

  useEffect(() => {
    (async () => {
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        console.log('Permission to access location was denied');
        setInjectedCode('true;');
        return;
      }

      try {
        let loc = await Location.getCurrentPositionAsync({});
        
        // This script runs BEFORE any website code, ensuring navigator.geolocation is ready instantly.
        const fakeGeolocation = `
          (function() {
            const locData = {
              coords: {
                latitude: ${loc.coords.latitude},
                longitude: ${loc.coords.longitude},
                accuracy: ${loc.coords.accuracy || 10},
                altitude: ${loc.coords.altitude || null},
                altitudeAccuracy: ${loc.coords.altitudeAccuracy || null},
                heading: ${loc.coords.heading || null},
                speed: ${loc.coords.speed || null}
              },
              timestamp: Date.now()
            };

            const overwriteGeolocation = () => {
              window.navigator.geolocation.getCurrentPosition = function(successCallback, errorCallback) {
                setTimeout(() => successCallback(locData), 0);
              };
              window.navigator.geolocation.watchPosition = function(successCallback) {
                setTimeout(() => successCallback(locData), 0);
                return 1;
              };
            };

            overwriteGeolocation();
            // Ensure child elements and third-party scripts utilize the mock coordinates
            Object.defineProperty(window.navigator, 'geolocation', {
              value: window.navigator.geolocation,
              configurable: false,
              writable: false
            });
          })();
          true;
        `;
        setInjectedCode(fakeGeolocation);
      } catch (error) {
        console.error("Error getting location:", error);
        setInjectedCode('true;');
      }
    })();
  }, []);

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" />
        {injectedCode ? (
          <WebView
            source={{ uri: PRODUCTION_URL }}
            style={styles.webview}
            javaScriptEnabled={true}
            domStorageEnabled={true}
            geolocationEnabled={true}
            injectedJavaScriptBeforeContentLoaded={injectedCode}
          />
        ) : (
          <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#0B1120' }}>
            <ActivityIndicator size="large" color="#0ea5e9" />
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
  }
});
