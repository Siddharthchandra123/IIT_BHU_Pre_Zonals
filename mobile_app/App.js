import React, { useState, useEffect } from 'react';
import { 
  StatusBar, 
  StyleSheet, 
  ActivityIndicator, 
  View, 
  Text, 
  TextInput, 
  TouchableOpacity, 
  ScrollView, 
  FlatList,
  Linking
} from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import * as Location from 'expo-location';

export default function App() {
  // Use your backend endpoint as before
  const BACKEND_URL = 'https://chikitsalya-backend.onrender.com/rag'; 
  const HOSPITAL_API = 'https://chikitsalya-backend.onrender.com/api/hospitals'; 
  
  const [activeTab, setActiveTab] = useState('assistant'); // 'assistant' or 'hospitals'
  const [loading, setLoading] = useState(false);
  const [location, setLocation] = useState(null);

  // Assistant State (Preserved logic)
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');

  // Hospital State
  const [hospitals, setHospitals] = useState([]);
  const [filteredHospitals, setFilteredHospitals] = useState([]);
  const [filter, setFilter] = useState('All');

  // Initialize Location & Fetch Hospitals
  useEffect(() => {
    (async () => {
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') return;

      const loc = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced });
      setLocation(loc);
      fetchHospitals(loc.coords.latitude, loc.coords.longitude);
    })();
  }, []);

  // --- PREDICTION MODEL (Untouched Logic) ---
  const handleQuery = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      // Exactly as you provided (sending { query }, reading data.answer)
      const response = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }), 
      });
      const data = await response.json();
      setAnswer(data.answer || 'No response from backend');
    } catch (error) {
      console.error(error);
      setAnswer('Error connecting to backend');
    } finally {
      setLoading(false);
    }
  };

  // Fetch Nearby Hospitals
  const fetchHospitals = async (lat, lon) => {
    try {
      const res = await fetch(`${HOSPITAL_API}?lat=${lat}&lon=${lon}`);
      const data = await res.json();
      const classified = data.map(h => classifyByName(h));
      setHospitals(classified);
      setFilteredHospitals(classified);
    } catch (err) {
      console.log("Hospital fetch error:", err);
    }
  };

  // --- CLASSIFICATION BY NAME (Requested Feature) ---
  const classifyByName = (facility) => {
    const name = facility.name.toLowerCase();
    let category = "Multispeciality";
    let icon = "🏥";

    if (name.includes('dent') || name.includes('tooth')) {
      category = "Dentist";
      icon = "🦷";
    } else if (name.includes('clinic') || name.includes('health care') || name.includes('dispensary') || name.includes('center')) {
      category = "Clinic";
      icon = "🩺";
    } else if (name.includes('private') || name.includes('pvt')) {
      category = "Private Checkup";
      icon = "🩺";
    } else if (name.includes('surgeon') || name.includes('surgery')) {
      category = "Surgeon";
      icon = "🔬";
    } else if (name.includes('hospital') || name.includes('medical college')) {
      category = "Multispeciality";
      icon = "🏥";
    }

    return { ...facility, category, icon };
  };

  // Client-side Filter Logic
  const handleFilter = (cat) => {
    setFilter(cat);
    if (cat === 'All') {
      setFilteredHospitals(hospitals);
    } else {
      setFilteredHospitals(hospitals.filter(h => h.category === cat));
    }
  };

  // Hospital Card Component
  const HospitalCard = ({ item }) => (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <View style={styles.iconBox}><Text style={{fontSize: 20}}>{item.icon || '🏥'}</Text></View>
        <View style={{ flex: 1 }}>
          <Text style={styles.cardTitle}>{item.name}</Text>
          <Text style={styles.cardAddress}>{item.address}</Text>
        </View>
        <View style={styles.categoryBadge}><Text style={styles.categoryText}>{item.category}</Text></View>
      </View>
      <View style={styles.cardActions}>
        {item.phone ? (
          <TouchableOpacity 
            style={styles.actionBtnCall} 
            onPress={() => Linking.openURL(`tel:${item.phone}`)}
          >
            <Text style={styles.btnText}>📞 Call</Text>
          </TouchableOpacity>
        ) : null}
        <TouchableOpacity 
          style={styles.actionBtnNav} 
          onPress={() => Linking.openURL(`https://maps.google.com/?q=${item.lat},${item.lon}`)}
        >
          <Text style={styles.btnText}>🗺️ Nav</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" />
        
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>⚕️ Chikitsalya AI</Text>
          <Text style={styles.headerSubtitle}>Medical Decision Support Tool</Text>
        </View>

        {/* Tab Selector */}
        <View style={styles.tabContainer}>
          <TouchableOpacity 
            style={[styles.tab, activeTab === 'assistant' && styles.activeTab]} 
            onPress={() => setActiveTab('assistant')}
          >
            <Text style={[styles.tabText, activeTab === 'assistant' && styles.activeTabText]}>Assistant</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.tab, activeTab === 'hospitals' && styles.activeTab]} 
            onPress={() => setActiveTab('hospitals')}
          >
            <Text style={[styles.tabText, activeTab === 'hospitals' && styles.activeTabText]}>Hospitals</Text>
          </TouchableOpacity>
        </View>

        {/* Dynamic Content */}
        <View style={{ flex: 1 }}>
          {activeTab === 'assistant' ? (
            <ScrollView contentContainerStyle={styles.content}>
              <View style={styles.inputBox}>
                <Text style={styles.label}>Ask something / symptoms:</Text>
                <TextInput 
                  style={styles.input} 
                  placeholder="Ask something..." 
                  placeholderTextColor="#64748b"
                  value={query}
                  onChangeText={setQuery}
                  multiline
                />
                <TouchableOpacity style={styles.predictBtn} onPress={handleQuery} disabled={loading}>
                  {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.btnTextMain}>Submit 🧠</Text>}
                </TouchableOpacity>
              </View>

              {answer ? (
                <View style={styles.resultBox}>
                  <Text style={styles.resultTitle}>Assistant Response:</Text>
                  <Text style={styles.resultText}>{answer}</Text>
                </View>
              ) : null}
            </ScrollView>
          ) : (
            <View style={{ flex: 1 }}>
              {/* Category Filters */}
              <View style={{ height: 60 }}>
                <ScrollView 
                  horizontal 
                  showsHorizontalScrollIndicator={false} 
                  contentContainerStyle={styles.filterBar}
                >
                  {['All', 'Multispeciality', 'Clinic', 'Dentist', 'Private Checkup', 'Surgeon'].map(cat => (
                    <TouchableOpacity 
                      key={cat} 
                      style={[styles.filterPill, filter === cat && styles.activePill]} 
                      onPress={() => handleFilter(cat)}
                    >
                      <Text style={[styles.pillText, filter === cat && styles.activePillText]}>
                        {cat === 'All' ? '🌍 All' : cat}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </ScrollView>
              </View>

              <FlatList
                data={filteredHospitals}
                keyExtractor={(item, index) => index.toString()}
                renderItem={HospitalCard}
                contentContainerStyle={{ padding: 15 }}
                ListEmptyComponent={() => (
                  <View style={styles.emptyBox}>
                    <Text style={styles.emptyText}>Searching for medical facilities... 📡</Text>
                  </View>
                )}
              />
            </View>
          )}
        </View>

      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0B1120' },
  header: { padding: 25, alignItems: 'center', borderBottomWidth: 1, borderBottomColor: '#1e293b' },
  headerTitle: { fontSize: 24, fontWeight: '800', color: '#0ea5e9' },
  headerSubtitle: { fontSize: 12, color: '#64748b', marginTop: 4 },
  
  tabContainer: { flexDirection: 'row', backgroundColor: '#162033', margin: 15, borderRadius: 12, padding: 4 },
  tab: { flex: 1, paddingVertical: 12, alignItems: 'center', borderRadius: 8 },
  activeTab: { backgroundColor: '#0ea5e9' },
  tabText: { color: '#94a3b8', fontWeight: '600' },
  activeTabText: { color: '#fff' },

  content: { padding: 20 },
  label: { color: '#94a3b8', fontSize: 14, marginBottom: 8, fontWeight: '500' },
  inputBox: { backgroundColor: '#1e293b', padding: 15, borderRadius: 16, borderWidth: 1, borderColor: '#334155' },
  input: { color: '#fff', fontSize: 16, minHeight: 80, textAlignVertical: 'top' },
  predictBtn: { backgroundColor: '#0ea5e9', marginTop: 15, padding: 15, borderRadius: 12, alignItems: 'center' },
  btnTextMain: { color: '#fff', fontSize: 16, fontWeight: '700' },
  resultBox: { marginTop: 25, backgroundColor: '#162033', padding: 20, borderRadius: 16, borderLeftWidth: 4, borderLeftColor: '#0ea5e9' },
  resultTitle: { color: '#0ea5e9', fontWeight: '700', fontSize: 12, textTransform: 'uppercase', marginBottom: 10 },
  resultText: { color: '#cbd5e1', fontSize: 15, lineHeight: 22 },

  filterBar: { paddingHorizontal: 15, alignItems: 'center', gap: 10 },
  filterPill: { paddingHorizontal: 20, paddingVertical: 8, borderRadius: 20, backgroundColor: '#1e293b', borderWidth: 1, borderColor: '#334155' },
  activePill: { backgroundColor: 'rgba(14, 165, 233, 0.2)', borderColor: '#0ea5e9' },
  pillText: { color: '#94a3b8', fontSize: 13, fontWeight: '600' },
  activePillText: { color: '#0ea5e9' },

  card: { backgroundColor: '#1e293b', borderRadius: 16, padding: 18, marginBottom: 15, borderWidth: 1, borderColor: '#334155' },
  cardHeader: { flexDirection: 'row', alignItems: 'flex-start', gap: 12 },
  iconBox: { width: 45, height: 45, backgroundColor: '#162033', borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  cardTitle: { color: '#fff', fontSize: 16, fontWeight: '700', flexShrink: 1 },
  cardAddress: { color: '#94a3b8', fontSize: 12, marginTop: 4, lineHeight: 16 },
  categoryBadge: { position: 'absolute', top: 0, right: 0, backgroundColor: 'rgba(14, 165, 233, 0.1)', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 20 },
  categoryText: { color: '#0ea5e9', fontSize: 10, fontWeight: '700' },
  cardActions: { flexDirection: 'row', gap: 10, marginTop: 15, paddingTop: 15, borderTopWidth: 1, borderTopColor: '#334155' },
  actionBtnCall: { flex: 1, padding: 10, backgroundColor: '#0f172a', borderRadius: 8, alignItems: 'center' },
  actionBtnNav: { flex: 1, padding: 10, backgroundColor: '#0ea5e9', borderRadius: 8, alignItems: 'center' },
  btnText: { color: '#fff', fontSize: 13, fontWeight: '600' },

  emptyBox: { alignItems: 'center', padding: 50 },
  emptyText: { color: '#64748b', fontSize: 14 }
});

