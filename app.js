/**
 * CropPulse AI — Frontend Application Logic
 * Implements two-way synchronized inputs, dual-engine ML prediction,
 * live weather geo-autofill, interactive crop encyclopedia, and theme toggling.
 */

// Crop Visual Badges & Category Mapping
const CROP_ICONS = {
  rice: '🍚',
  maize: '🌽',
  chickpea: '🧆',
  kidneybeans: '🫘',
  pigeonpeas: '🌱',
  mothbeans: '🌿',
  mungbean: '🥗',
  blackgram: '🥣',
  lentil: '🍲',
  pomegranate: '🍎',
  banana: '🍌',
  mango: '🥭',
  grapes: '🍇',
  watermelon: '🍉',
  muskmelon: '🍈',
  apple: '🍏',
  orange: '🍊',
  papaya: '🍈',
  coconut: '🥥',
  cotton: '🌿',
  jute: '🌾',
  coffee: '☕'
};

// Preset Scenarios
const PRESET_SCENARIOS = {
  monsoon_rice: { N: 90, P: 42, K: 43, temp: 24.5, humidity: 82.0, ph: 6.5, rainfall: 240.0, name: "Monsoon Rice" },
  highland_coffee: { N: 105, P: 28, K: 30, temp: 25.5, humidity: 58.0, ph: 6.8, rainfall: 160.0, name: "Highland Coffee" },
  black_cotton: { N: 118, P: 45, K: 20, temp: 24.0, humidity: 80.0, ph: 6.8, rainfall: 80.0, name: "Black Soil Cotton" },
  golden_jute: { N: 80, P: 48, K: 40, temp: 25.0, humidity: 80.0, ph: 6.7, rainfall: 175.0, name: "Golden Jute" },
  temperate_apple: { N: 20, P: 135, K: 200, temp: 22.5, humidity: 92.0, ph: 6.0, rainfall: 110.0, name: "Temperate Apple" },
  tropical_banana: { N: 100, P: 82, K: 50, temp: 27.5, humidity: 80.0, ph: 6.0, rainfall: 105.0, name: "Tropical Banana" },
  coastal_coconut: { N: 22, P: 18, K: 30, temp: 27.0, humidity: 95.0, ph: 6.2, rainfall: 175.0, name: "Coastal Coconut" },
  summer_watermelon: { N: 100, P: 18, K: 50, temp: 26.0, humidity: 85.0, ph: 6.5, rainfall: 50.0, name: "Summer Watermelon" },
  dryland_chickpea: { N: 40, P: 68, K: 79, temp: 19.0, humidity: 17.0, ph: 7.3, rainfall: 80.0, name: "Dryland Chickpea" },
  fertile_maize: { N: 80, P: 45, K: 20, temp: 23.0, humidity: 65.0, ph: 6.2, rainfall: 70.0, name: "Fertile Maize" }
};

// Global state
let allCropsCatalog = {};
let clientTreeModel = null;

// DOM Elements
const elements = {
  // Inputs
  sliderN: document.getElementById('slider-N'),
  inputN: document.getElementById('input-N'),
  sliderP: document.getElementById('slider-P'),
  inputP: document.getElementById('input-P'),
  sliderK: document.getElementById('slider-K'),
  inputK: document.getElementById('input-K'),
  sliderTemp: document.getElementById('slider-temp'),
  inputTemp: document.getElementById('input-temp'),
  sliderHumidity: document.getElementById('slider-humidity'),
  inputHumidity: document.getElementById('input-humidity'),
  sliderPh: document.getElementById('slider-ph'),
  inputPh: document.getElementById('input-ph'),
  sliderRainfall: document.getElementById('slider-rainfall'),
  inputRainfall: document.getElementById('input-rainfall'),
  phStatusLabel: document.getElementById('ph-status-label'),

  // Actions
  btnPredict: document.getElementById('btn-predict'),
  btnReset: document.getElementById('btn-reset'),
  btnRandomize: document.getElementById('btn-randomize'),
  btnFetchWeather: document.getElementById('btn-fetch-weather'),
  btnGeolocation: document.getElementById('btn-geolocation'),
  weatherCityInput: document.getElementById('weather-city-input'),
  btnPrintReport: document.getElementById('btn-print-report'),

  // Results
  resultCard: document.getElementById('result-card'),
  resultIcon: document.getElementById('result-icon'),
  resultCategory: document.getElementById('result-category'),
  resultName: document.getElementById('result-name'),
  resultConfidence: document.getElementById('result-confidence'),
  resultProgress: document.getElementById('result-progress'),
  resultSeason: document.getElementById('result-season'),
  resultDuration: document.getElementById('result-duration'),
  resultWater: document.getElementById('result-water'),
  resultSoil: document.getElementById('result-soil'),
  resultDescription: document.getElementById('result-description'),
  resultFertilizer: document.getElementById('result-fertilizer'),
  resultAlignmentList: document.getElementById('result-alignment-list'),
  resultAlternatesFlex: document.getElementById('result-alternates-flex'),

  // Tabs
  tabBtns: document.querySelectorAll('.nav-tab-btn'),
  tabContents: document.querySelectorAll('.tab-content'),

  // Encyclopedia
  encyclopediaGrid: document.getElementById('encyclopedia-grid'),
  cropSearchInput: document.getElementById('crop-search-input'),
  catFilterBtns: document.querySelectorAll('.cat-filter-btn'),

  // Theme
  themeToggle: document.getElementById('theme-toggle'),
  toastContainer: document.getElementById('toast-container')
};

// ==================== INITIALIZATION ====================

document.addEventListener('DOMContentLoaded', () => {
  setupInputSync();
  setupEventListeners();
  setupTabs();
  initTheme();
  loadCropCatalog();
  loadClientTreeModel();
  
  // Run initial prediction for default values
  triggerPrediction();
});

// ==================== INPUT SYNCHRONIZATION ====================

function setupInputSync() {
  const pairs = [
    { slider: elements.sliderN, input: elements.inputN },
    { slider: elements.sliderP, input: elements.inputP },
    { slider: elements.sliderK, input: elements.inputK },
    { slider: elements.sliderTemp, input: elements.inputTemp },
    { slider: elements.sliderHumidity, input: elements.inputHumidity },
    { slider: elements.sliderPh, input: elements.inputPh, isPh: true },
    { slider: elements.sliderRainfall, input: elements.inputRainfall }
  ];

  pairs.forEach(({ slider, input, isPh }) => {
    if (!slider || !input) return;

    slider.addEventListener('input', () => {
      input.value = slider.value;
      updateSliderTrack(slider);
      if (isPh) updatePhIndicator(slider.value);
    });

    input.addEventListener('input', () => {
      let val = parseFloat(input.value);
      if (isNaN(val)) return;
      slider.value = val;
      updateSliderTrack(slider);
      if (isPh) updatePhIndicator(val);
    });

    updateSliderTrack(slider);
  });

  updatePhIndicator(elements.sliderPh.value);
}

function updateSliderTrack(slider) {
  const min = parseFloat(slider.min) || 0;
  const max = parseFloat(slider.max) || 100;
  const val = parseFloat(slider.value) || 0;
  const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
  slider.style.background = `linear-gradient(to right, #10b981 0%, #10b981 ${pct}%, var(--bg-elevated) ${pct}%, var(--bg-elevated) 100%)`;
}

function updatePhIndicator(phVal) {
  if (!elements.phStatusLabel) return;
  const val = parseFloat(phVal);
  if (val < 5.5) {
    elements.phStatusLabel.textContent = "Strongly Acidic (Requires Liming)";
    elements.phStatusLabel.style.color = "#ef4444";
  } else if (val < 6.5) {
    elements.phStatusLabel.textContent = "Moderately Acidic (Good for Tea/Berries)";
    elements.phStatusLabel.style.color = "#f59e0b";
  } else if (val <= 7.5) {
    elements.phStatusLabel.textContent = "Neutral / Optimal (Ideal for Most Crops)";
    elements.phStatusLabel.style.color = "#10b981";
  } else if (val <= 8.5) {
    elements.phStatusLabel.textContent = "Moderately Alkaline (Good for Cotton/Legumes)";
    elements.phStatusLabel.style.color = "#38bdf8";
  } else {
    elements.phStatusLabel.textContent = "Strongly Alkaline (High Salinity Risk)";
    elements.phStatusLabel.style.color = "#8b5cf6";
  }
}

// ==================== EVENT LISTENERS ====================

function setupEventListeners() {
  // Predict button
  elements.btnPredict.addEventListener('click', () => {
    triggerPrediction();
  });

  // Reset button
  elements.btnReset.addEventListener('click', () => {
    setFormValues(PRESET_SCENARIOS.monsoon_rice);
    triggerPrediction();
    showToast("Reset inputs to standard baseline.");
  });

  // Randomize button
  elements.btnRandomize.addEventListener('click', () => {
    randomizeValues();
  });

  // Preset chips
  document.querySelectorAll('.preset-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const presetId = chip.getAttribute('data-preset');
      if (PRESET_SCENARIOS[presetId]) {
        setFormValues(PRESET_SCENARIOS[presetId]);
        triggerPrediction();
        showToast(`Loaded preset: ${PRESET_SCENARIOS[presetId].name}`);
      }
    });
  });

  // Weather fetch
  elements.btnFetchWeather.addEventListener('click', () => {
    const city = elements.weatherCityInput.value.trim();
    if (!city) {
      showToast("Please enter a city name first!", "warning");
      return;
    }
    fetchWeatherByCity(city);
  });

  elements.weatherCityInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      elements.btnFetchWeather.click();
    }
  });

  // Geolocation
  elements.btnGeolocation.addEventListener('click', () => {
    if (!navigator.geolocation) {
      showToast("Geolocation is not supported by your browser.", "warning");
      return;
    }
    showToast("Detecting GPS coordinates...", "info");
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        fetchWeatherByCoords(pos.coords.latitude, pos.coords.longitude, "Your Location");
      },
      (err) => {
        showToast("Location access denied or timed out.", "warning");
      }
    );
  });

  // Print Report
  elements.btnPrintReport.addEventListener('click', () => {
    window.print();
  });

  // Theme Toggle
  elements.themeToggle.addEventListener('click', () => {
    toggleTheme();
  });

  // Encyclopedia search & filters
  if (elements.cropSearchInput) {
    elements.cropSearchInput.addEventListener('input', () => {
      filterEncyclopedia();
    });
  }

  elements.catFilterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.catFilterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      filterEncyclopedia();
    });
  });
}

function setFormValues(p) {
  elements.sliderN.value = p.N; elements.inputN.value = p.N;
  elements.sliderP.value = p.P; elements.inputP.value = p.P;
  elements.sliderK.value = p.K; elements.inputK.value = p.K;
  elements.sliderTemp.value = p.temp; elements.inputTemp.value = p.temp;
  elements.sliderHumidity.value = p.humidity; elements.inputHumidity.value = p.humidity;
  elements.sliderPh.value = p.ph; elements.inputPh.value = p.ph;
  elements.sliderRainfall.value = p.rainfall; elements.inputRainfall.value = p.rainfall;

  [elements.sliderN, elements.sliderP, elements.sliderK, elements.sliderTemp, elements.sliderHumidity, elements.sliderPh, elements.sliderRainfall].forEach(s => updateSliderTrack(s));
  updatePhIndicator(p.ph);
}

function randomizeValues() {
  const presets = Object.values(PRESET_SCENARIOS);
  const randomPreset = presets[Math.floor(Math.random() * presets.length)];
  // Add subtle jitter
  const jitter = (val, spread) => Math.max(5, Math.round((val + (Math.random() * spread * 2 - spread)) * 10) / 10);
  
  const randomized = {
    N: Math.min(140, jitter(randomPreset.N, 8)),
    P: Math.min(140, jitter(randomPreset.P, 6)),
    K: Math.min(200, jitter(randomPreset.K, 8)),
    temp: Math.min(42, Math.max(12, jitter(randomPreset.temp, 2))),
    humidity: Math.min(98, Math.max(20, jitter(randomPreset.humidity, 5))),
    ph: Math.min(9.0, Math.max(4.5, jitter(randomPreset.ph, 0.4))),
    rainfall: Math.min(280, Math.max(30, jitter(randomPreset.rainfall, 15))),
    name: "Random Agro-Scenario"
  };
  
  setFormValues(randomized);
  triggerPrediction();
  showToast("Generated realistic randomized conditions.");
}

// ==================== PREDICTION ENGINE ====================

async function triggerPrediction() {
  elements.btnPredict.classList.add('loading');
  elements.btnPredict.disabled = true;

  const payload = {
    N: parseFloat(elements.inputN.value),
    P: parseFloat(elements.inputP.value),
    K: parseFloat(elements.inputK.value),
    temperature: parseFloat(elements.inputTemp.value),
    humidity: parseFloat(elements.inputHumidity.value),
    ph: parseFloat(elements.inputPh.value),
    rainfall: parseFloat(elements.inputRainfall.value)
  };

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (response.ok) {
      const data = await response.json();
      renderPredictionResult(data);
    } else {
      // Fallback to client-side pure tree inference
      runClientPrediction(payload);
    }
  } catch (err) {
    console.warn("API request failed, running client-side fallback inference:", err);
    runClientPrediction(payload);
  } finally {
    elements.btnPredict.classList.remove('loading');
    elements.btnPredict.disabled = false;
  }
}

function renderPredictionResult(data) {
  const p = data.prediction;
  const cropId = p.crop.toLowerCase();
  
  // Icon & Header
  elements.resultIcon.textContent = CROP_ICONS[cropId] || '🌾';
  elements.resultName.textContent = p.name;
  elements.resultCategory.textContent = p.category;

  // Confidence & Suitability
  const confText = `${p.confidence}% Match`;
  elements.resultConfidence.textContent = confText;
  elements.resultProgress.style.width = `${Math.min(100, p.confidence)}%`;

  // Specifications
  elements.resultSeason.textContent = p.season || "Seasonal";
  elements.resultDuration.textContent = p.growth_duration || "90 - 120 days";
  elements.resultWater.textContent = p.water_requirement || "Moderate";
  elements.resultSoil.textContent = p.soil_type || "Loam / Sandy Loam";

  // Description & Fertilizer
  elements.resultDescription.textContent = p.description;
  elements.resultFertilizer.textContent = p.fertilizer_guide || "Standard balanced NPK application.";

  // Nutrient Diagnostics List
  renderDiagnostics(data.suitability_analysis);

  // Alternates
  renderAlternates(data.alternates);

  // Pulse animation on result card
  elements.resultCard.classList.remove('has-result');
  void elements.resultCard.offsetWidth; // Trigger reflow
  elements.resultCard.classList.add('has-result');
}

function renderDiagnostics(analysis) {
  if (!elements.resultAlignmentList || !analysis) return;
  elements.resultAlignmentList.innerHTML = '';

  const paramOrder = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'];
  paramOrder.forEach(key => {
    const item = analysis[key];
    if (!item) return;

    const div = document.createElement('div');
    div.className = 'alignment-item';

    let badgeClass = 'badge-optimal';
    if (item.status === 'Low') badgeClass = 'badge-warning';
    if (item.status === 'High') badgeClass = 'badge-danger';

    div.innerHTML = `
      <div>
        <span class="alignment-item-param">${item.name} (${item.user_value} ${item.unit})</span>
        <div style="font-size: 0.73rem; color: var(--text-muted); margin-top: 2px;">${item.advisory}</div>
      </div>
      <span class="alignment-status-badge ${badgeClass}">${item.status}</span>
    `;
    elements.resultAlignmentList.appendChild(div);
  });
}

function renderAlternates(alternates) {
  if (!elements.resultAlternatesFlex) return;
  elements.resultAlternatesFlex.innerHTML = '';

  if (!alternates || alternates.length === 0) {
    elements.resultAlternatesFlex.innerHTML = '<span style="font-size: 0.8rem; color: var(--text-muted);">No close secondary alternatives found.</span>';
    return;
  }

  alternates.forEach(alt => {
    const icon = CROP_ICONS[alt.crop.toLowerCase()] || '🌱';
    const pill = document.createElement('div');
    pill.className = 'alternate-pill';
    pill.innerHTML = `${icon} ${alt.name} <span class="alt-conf">(${alt.confidence}%)</span>`;
    pill.addEventListener('click', () => {
      // Find preset or load details
      if (allCropsCatalog[alt.crop]) {
        loadCropIntoPredictor(alt.crop);
      }
    });
    elements.resultAlternatesFlex.appendChild(pill);
  });
}

// ==================== CLIENT-SIDE FALLBACK INFERENCE ====================

async function loadClientTreeModel() {
  try {
    const res = await fetch('model_data.json');
    if (res.ok) {
      clientTreeModel = await res.json();
    }
  } catch (e) {
    // Silently continue
  }
}

function runClientPrediction(input) {
  if (!clientTreeModel) {
    showToast("Evaluating via default rules.", "info");
    return;
  }

  const features = [input.N, input.P, input.K, input.temperature, input.humidity, input.ph, input.rainfall];
  const classes = clientTreeModel.classes;
  const accum = new Array(classes.length).fill(0);
  const trees = clientTreeModel.trees;

  trees.forEach(tree => {
    let node = 0;
    while (tree.children_left[node] !== -1) {
      if (features[tree.feature[node]] <= tree.threshold[node]) {
        node = tree.children_left[node];
      } else {
        node = tree.children_right[node];
      }
    }
    const val = tree.value[node];
    const sum = val.reduce((a, b) => a + b, 0);
    if (sum > 0) {
      for (let i = 0; i < classes.length; i++) {
        accum[i] += val[i] / sum;
      }
    }
  });

  const probas = accum.map((p, i) => ({ crop: classes[i], prob: p / trees.length }));
  probas.sort((a, b) => b.prob - a.prob);

  const top = probas[0];
  const cropId = top.crop.toLowerCase();
  const meta = allCropsCatalog[cropId] || {};

  renderPredictionResult({
    prediction: {
      crop: cropId,
      name: meta.common_name || top.crop.toUpperCase(),
      confidence: Math.round(top.prob * 1000) / 10,
      category: meta.category || "Agricultural Crop",
      season: meta.season || "Seasonal",
      growth_duration: meta.growth_duration || "90-120 days",
      water_requirement: meta.water_requirement || "Moderate",
      soil_type: meta.soil_type || "Loam",
      fertilizer_guide: meta.fertilizer_guide || "Standard NPK",
      description: meta.description || `${top.crop} is optimal for current conditions.`
    },
    alternates: probas.slice(1, 4).filter(x => x.prob > 0.01).map(x => ({
      crop: x.crop,
      name: (allCropsCatalog[x.crop] && allCropsCatalog[x.crop].common_name) || x.crop,
      confidence: Math.round(x.prob * 1000) / 10
    })),
    suitability_analysis: {}
  });
}

// ==================== LIVE WEATHER GEO-AUTOFILL ====================

async function fetchWeatherByCity(cityName) {
  showToast(`Searching coordinates for ${cityName}...`, "info");
  try {
    const geoRes = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(cityName)}&count=1&language=en&format=json`);
    const geoData = await geoRes.json();
    if (!geoData.results || geoData.results.length === 0) {
      showToast(`City "${cityName}" not found. Try another city.`, "warning");
      return;
    }

    const { latitude, longitude, name, country } = geoData.results[0];
    await fetchWeatherByCoords(latitude, longitude, `${name}, ${country || ''}`);
  } catch (err) {
    showToast("Weather service error. Check connection.", "warning");
  }
}

async function fetchWeatherByCoords(lat, lon, label) {
  try {
    const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m&daily=precipitation_sum&timezone=auto`;
    const res = await fetch(url);
    const data = await res.json();

    if (data && data.current) {
      const temp = Math.round(data.current.temperature_2m * 10) / 10;
      const humidity = Math.round(data.current.relative_humidity_2m * 10) / 10;

      // Estimate seasonal rainfall based on local climate
      let rainfall = parseFloat(elements.sliderRainfall.value);
      if (data.daily && data.daily.precipitation_sum) {
        const sum = data.daily.precipitation_sum.reduce((a, b) => a + b, 0);
        rainfall = Math.max(30, Math.min(280, Math.round(sum * 25 + 60)));
      }

      elements.sliderTemp.value = temp;
      elements.inputTemp.value = temp;
      elements.sliderHumidity.value = humidity;
      elements.inputHumidity.value = humidity;
      elements.sliderRainfall.value = rainfall;
      elements.inputRainfall.value = rainfall;

      updateSliderTrack(elements.sliderTemp);
      updateSliderTrack(elements.sliderHumidity);
      updateSliderTrack(elements.sliderRainfall);

      triggerPrediction();
      showToast(`☀️ Updated climate for ${label}: ${temp}°C, ${humidity}% humidity!`);
    }
  } catch (err) {
    showToast("Could not retrieve live meteorological data.", "warning");
  }
}

// ==================== CROP ENCYCLOPEDIA ====================

async function loadCropCatalog() {
  try {
    const res = await fetch('crop_metadata.json');
    if (res.ok) {
      allCropsCatalog = await res.json();
      renderEncyclopediaCards(allCropsCatalog);
    }
  } catch (e) {
    console.warn("Could not load crop_metadata.json:", e);
  }
}

function renderEncyclopediaCards(cropsObj) {
  if (!elements.encyclopediaGrid) return;
  elements.encyclopediaGrid.innerHTML = '';

  Object.keys(cropsObj).forEach(key => {
    const c = cropsObj[key];
    const icon = CROP_ICONS[key] || '🌱';
    const card = document.createElement('div');
    card.className = 'crop-encyclo-card';
    card.setAttribute('data-name', (c.common_name || key).toLowerCase());
    card.setAttribute('data-category', c.category || '');

    const stats = c.stats || {};
    const nMean = stats.N ? stats.N.mean : '-';
    const pMean = stats.P ? stats.P.mean : '-';
    const kMean = stats.K ? stats.K.mean : '-';
    const tempMean = stats.temperature ? stats.temperature.mean : '-';
    const rainMean = stats.rainfall ? stats.rainfall.mean : '-';

    card.innerHTML = `
      <div class="card-avatar-row">
        <span class="card-avatar-icon">${icon}</span>
        <span class="crop-hero-badge">${c.category || 'Crop'}</span>
      </div>
      <h3 class="crop-card-name">${c.common_name || key.toUpperCase()}</h3>
      <p class="crop-card-desc">${c.description || ''}</p>
      
      <div class="crop-param-pills">
        <div>N-P-K<span>${nMean}/${pMean}/${kMean}</span></div>
        <div>Temp<span>${tempMean}°C</span></div>
        <div>Rain<span>${rainMean}mm</span></div>
      </div>
      
      <button class="card-load-btn" onclick="loadCropIntoPredictor('${key}')">
        ⚡ Load into Predictor
      </button>
    `;
    elements.encyclopediaGrid.appendChild(card);
  });
}

window.loadCropIntoPredictor = function(cropKey) {
  const crop = allCropsCatalog[cropKey];
  if (!crop || !crop.stats) return;

  const s = crop.stats;
  setFormValues({
    N: s.N.mean,
    P: s.P.mean,
    K: s.K.mean,
    temp: s.temperature.mean,
    humidity: s.humidity.mean,
    ph: s.ph.mean,
    rainfall: s.rainfall.mean,
    name: crop.common_name
  });

  // Switch to Predictor tab
  document.getElementById('tab-btn-predictor').click();
  triggerPrediction();
  showToast(`Loaded optimal benchmarks for ${crop.common_name || cropKey}!`);
};

function filterEncyclopedia() {
  const query = elements.cropSearchInput.value.toLowerCase().trim();
  const activeCatBtn = document.querySelector('.cat-filter-btn.active');
  const cat = activeCatBtn ? activeCatBtn.getAttribute('data-category') : 'all';

  const cards = document.querySelectorAll('.crop-encyclo-card');
  cards.forEach(card => {
    const name = card.getAttribute('data-name');
    const cardCat = card.getAttribute('data-category');

    const matchesQuery = !query || name.includes(query);
    const matchesCat = (cat === 'all') || cardCat.includes(cat);

    if (matchesQuery && matchesCat) {
      card.style.display = 'flex';
    } else {
      card.style.display = 'none';
    }
  });
}

// ==================== TABS SYSTEM ====================

function setupTabs() {
  elements.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.tabBtns.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      elements.tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      const tabId = btn.getAttribute('data-tab');
      const targetContent = document.getElementById(tabId);
      if (targetContent) {
        targetContent.classList.add('active');
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });
  });
}

// ==================== THEME MANAGEMENT ====================

function initTheme() {
  const savedTheme = localStorage.getItem('croppulse_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('croppulse_theme', next);
  updateThemeIcon(next);
}

function updateThemeIcon(theme) {
  elements.themeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
}

// ==================== TOAST NOTIFICATIONS ====================

function showToast(message, type = 'success') {
  if (!elements.toastContainer) return;

  const toast = document.createElement('div');
  toast.className = 'toast';
  
  let icon = '🌱';
  if (type === 'warning') icon = '⚠️';
  if (type === 'info') icon = 'ℹ️';

  toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
  elements.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3200);
}

// ==================== STEPPER INPUT HELPER ====================

window.stepInput = function(param, delta) {
  let input, slider;
  if (param === 'N') { input = elements.inputN; slider = elements.sliderN; }
  else if (param === 'P') { input = elements.inputP; slider = elements.sliderP; }
  else if (param === 'K') { input = elements.inputK; slider = elements.sliderK; }
  else if (param === 'temp') { input = elements.inputTemp; slider = elements.sliderTemp; }
  else if (param === 'humidity') { input = elements.inputHumidity; slider = elements.sliderHumidity; }
  else if (param === 'ph') { input = elements.inputPh; slider = elements.sliderPh; }
  else if (param === 'rainfall') { input = elements.inputRainfall; slider = elements.sliderRainfall; }

  if (!input || !slider) return;

  const min = parseFloat(slider.min);
  const max = parseFloat(slider.max);
  let current = parseFloat(input.value) || min;
  let next = Math.round((current + delta) * 10) / 10;
  next = Math.max(min, Math.min(max, next));

  input.value = next;
  slider.value = next;
  updateSliderTrack(slider);
  if (param === 'ph') updatePhIndicator(next);

  // Trigger prediction update
  triggerPrediction();
};

