import { Text, SafeAreaView, StyleSheet } from 'react-native';

// You can import supported modules from npm
import { Card } from 'react-native-paper';

// or any files within the Snack
import CameraComponent from './components/camera'


export default function App() {
  return (
    
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>
        Take a picture below to categorize your waste!
      </Text>
      <CameraComponent>
      </CameraComponent>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    backgroundColor: '#ecf0f1',
    padding: 8,
  },
  title: {
    margin: 24,
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center'
  }
});
