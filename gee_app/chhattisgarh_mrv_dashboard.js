/**
 * ==============================================================================
 * CHHATTISGARH BLOCK-LEVEL RICE METHANE MRV & GEE INTERACTIVE DASHBOARD
 * ==============================================================================
 * Author: Akshit Vaishnav (akshitvaishnavonline@gmail.com)
 * Project: Satellite-Based Rice Methane Assessment & Carbon MRV Feasibility
 * Study Area: Chhattisgarh, India (EPSG:32644 / WGS84)
 * Platform: Google Earth Engine UI API
 *
 * INSTRUCTIONS FOR USE IN GEE CODE EDITOR:
 * 1. Open https://code.earthengine.google.com/
 * 2. Paste this entire script into the Code Editor.
 * 3. Ensure your active cloud project is set to 'ee-akshitvaishnavonline'.
 * 4. Click 'Run'.
 * ==============================================================================
 */

// ------------------------------------------------------------------------------
// 1. Spatial Domain & Agro-Ecological Boundaries
// ------------------------------------------------------------------------------
var cgState = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM1_NAME', 'Chhattisgarh'));

var cgDistricts = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM1_NAME', 'Chhattisgarh'));

var stateGeom = cgState.geometry();

// ------------------------------------------------------------------------------
// 2. Earth Engine Image Collections & Data Preparation
// ------------------------------------------------------------------------------
var currentYear = 2024;

function getKharifDates(year) {
  return {
    start: ee.Date.fromYMD(year, 6, 15),
    end: ee.Date.fromYMD(year, 11, 15)
  };
}

// Sentinel-1 SAR Dual-Pol Inundation Composite
function getSentinel1SAR(year) {
  var dates = getKharifDates(year);
  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(stateGeom)
    .filterDate(dates.start, dates.end)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'));

  var vv = s1.select('VV').median().clip(stateGeom);
  var vh = s1.select('VH').median().clip(stateGeom);
  var ratio = vh.subtract(vv).rename('VH_VV_ratio');

  // JRC Global Surface Water permanent water mask (occurrence > 80% = permanent water bodies/reservoirs)
  // Ensures permanent village tanks (Talabs) and reservoirs from the National Water Body Census
  // are excluded from the Sentinel-1 flood frequency and seasonal paddy inundation calculation.
  var jrcPermanentWater = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    .select('occurrence')
    .gt(80)
    .clip(stateGeom);

  // SAR Inundation: low VV (< -15 dB) indicating standing water / puddling, masked for permanent water
  var rawInundation = vv.lt(-15.0);
  var inundationMask = rawInundation.and(jrcPermanentWater.not()).rename('sar_inundation_mask');

  return vv.addBands(vh).addBands(ratio).addBands(inundationMask).addBands(jrcPermanentWater.rename('permanent_water_mask'));
}

// Sentinel-2 Optical MSI Cloud-Masked Phenology Composite
function getSentinel2MSI(year) {
  var dates = getKharifDates(year);

  function maskS2clouds(image) {
    var qa = image.select('QA60');
    var cloudBitMask = 1 << 10;
    var cirrusBitMask = 1 << 11;
    var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
    return image.updateMask(mask).divide(10000);
  }

  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(stateGeom)
    .filterDate(dates.start, dates.end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 25))
    .map(maskS2clouds)
    .median()
    .clip(stateGeom);

  var ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndwi = s2.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var lswi = s2.normalizedDifference(['B8', 'B11']).rename('LSWI');

  return s2.addBands(ndvi).addBands(ndwi).addBands(lswi);
}

// SoilGrids 250m v2.0 Properties
var soilClay = ee.Image('projects/soilgrids-isric/clay_mean')
  .select('clay_0-5cm_mean')
  .divide(10.0)
  .clip(stateGeom)
  .rename('clay_pct');

var soilSOC = ee.Image('projects/soilgrids-isric/soc_mean')
  .select('soc_0-5cm_mean')
  .divide(10.0)
  .clip(stateGeom)
  .rename('soc_g_kg');

// NASA SRTM DEM Topography
var srtm = ee.Image('USGS/SRTMGL1_003').clip(stateGeom);
var elevation = srtm.select('elevation');
var slope = ee.Terrain.slope(elevation).rename('slope');

// Modeled IPCC Tier-2 Rice Methane Flux Layer (kg CH4/ha/season)
function getMethaneFluxLayer(year) {
  var s1 = getSentinel1SAR(year);
  var s2 = getSentinel2MSI(year);

  // Inundation weight derived from SAR backscatter and optical LSWI
  var inundationScore = s1.select('sar_inundation_mask')
    .multiply(0.6)
    .add(s2.select('LSWI').gt(0.2).multiply(0.4));

  // Base IPCC emission factor: 1.30 kg/ha/day * 115 days ~ 149.5 kg/ha baseline
  var baseFlux = ee.Image.constant(149.5);

  // Water regime factor (continuous vs intermittent): 0.60 to 1.05
  var sfW = ee.Image.constant(0.60).add(inundationScore.multiply(0.45));

  // Organic amendments (FYM scaling factor): ~1.18
  var sfO = ee.Image.constant(1.18);

  // Soil modifier from ISRIC Clay & SOC
  var sfS = ee.Image.constant(0.90)
    .add(soilClay.divide(100.0).multiply(0.35))
    .add(soilSOC.divide(20.0).multiply(0.15));

  // Temperature / Topography modifier
  var sfT = ee.Image.constant(1.05).subtract(elevation.divide(2000.0).multiply(0.25));

  // Total Kharif Methane Flux (kg CH4 / ha / season)
  var ch4Flux = baseFlux.multiply(sfW).multiply(sfO).multiply(sfS).multiply(sfT)
    .clip(stateGeom)
    .rename('ch4_flux_kg_ha');

  // Mask out non-cultivated pixels (NDVI < 0.40 or water bodies)
  var riceMask = s2.select('NDVI').gt(0.45).and(slope.lt(8.0));
  var maskedFlux = ch4Flux.updateMask(riceMask);

  // Counterfactual AWD Mitigation Potential (38% reduction under Verra VM0051)
  var awdMitigation = maskedFlux.multiply(0.38).rename('awd_mitigation_kg_ha');

  return {
    flux: maskedFlux,
    awd: awdMitigation,
    s1: s1,
    s2: s2
  };
}

// ------------------------------------------------------------------------------
// 3. Visualization Palettes
// ------------------------------------------------------------------------------
var visFlux = {
  min: 80,
  max: 220,
  palette: ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']
};

var visAWD = {
  min: 25,
  max: 85,
  palette: ['#edf8fb', '#b2e2e2', '#66c2a4', '#2ca25f', '#006d2c']
};

var visSAR = {
  bands: ['VV', 'VH', 'VH_VV_ratio'],
  min: [-22, -28, -12],
  max: [-6, -10, -2]
};

var visS2FCC = {
  bands: ['B8', 'B4', 'B3'],
  min: 0.05,
  max: 0.45
};

// ------------------------------------------------------------------------------
// 4. User Interface Architecture
// ------------------------------------------------------------------------------
ui.root.clear();

// Left Panel for Analytics & Controls
var panel = ui.Panel({
  style: {
    width: '380px',
    padding: '12px',
    backgroundColor: '#f8f9fa'
  }
});

// Map Viewport
var map = ui.Map();
map.setCenter(81.8, 21.2, 7);
map.setOptions('HYBRID');

ui.root.add(panel);
ui.root.add(map);

// Title & Meta Info
var title = ui.Label({
  value: '🌾 Chhattisgarh Rice Methane MRV Dashboard',
  style: {fontSize: '18px', fontWeight: 'bold', color: '#1b5e20', margin: '0 0 4px 0'}
});

var subtitle = ui.Label({
  value: 'Sentinel-1 SAR + Sentinel-2 MSI + SoilGrids + IPCC Tier-2 Ensemble',
  style: {fontSize: '11px', color: '#555', margin: '0 0 12px 0'}
});

panel.add(title);
panel.add(subtitle);

// Year Selector Dropdown
var yearLabel = ui.Label('1. Select Analysis Season (2019 - 2026):', {fontWeight: 'bold', fontSize: '12px'});
panel.add(yearLabel);

var yearSelect = ui.Select({
  items: ['2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026'],
  value: '2024',
  onChange: function(selectedYear) {
    currentYear = parseInt(selectedYear);
    updateMapLayers();
  },
  style: {width: '95%', margin: '0 0 10px 0'}
});
panel.add(yearSelect);

// District Quick-Zoom Dropdown
var zoomLabel = ui.Label('2. Zoom to Key Agricultural Basin:', {fontWeight: 'bold', fontSize: '12px'});
panel.add(zoomLabel);

var districtCoords = {
  'Entire Chhattisgarh': [81.8, 21.2, 7],
  'Durg (Validation Basin)': [81.28, 21.19, 10],
  'Raipur (Central Plains)': [81.63, 21.25, 10],
  'Bilaspur (Northern Plains)': [82.14, 22.08, 10],
  'Bastar / Jagdalpur (Southern Plateau)': [81.95, 19.08, 9],
  'Surguja / Ambikapur (Northern Hills)': [83.19, 23.12, 9]
};

var districtSelect = ui.Select({
  items: Object.keys(districtCoords),
  value: 'Entire Chhattisgarh',
  onChange: function(key) {
    var coords = districtCoords[key];
    map.setCenter(coords[0], coords[1], coords[2]);
  },
  style: {width: '95%', margin: '0 0 12px 0'}
});
panel.add(districtSelect);

// Layer Toggles
var layerSectionLabel = ui.Label('3. Active Thematic Layers:', {fontWeight: 'bold', fontSize: '12px'});
panel.add(layerSectionLabel);

var toggleMethane = ui.Checkbox('IPCC Tier-2 Methane Flux (kg CH₄/ha)', true);
var toggleAWD = ui.Checkbox('AWD Mitigation Potential (Verra VM0051)', false);
var toggleSAR = ui.Checkbox('Sentinel-1 SAR Inundation Composite', false);
var toggleS2 = ui.Checkbox('Sentinel-2 Optical False Color (B8/B4/B3)', false);
var toggleDistricts = ui.Checkbox('District Boundaries (33 Districts)', true);

[toggleMethane, toggleAWD, toggleSAR, toggleS2, toggleDistricts].forEach(function(cb) {
  cb.onChange(function() { updateMapLayers(); });
  panel.add(cb);
});

// Inspector Card
var inspectorTitle = ui.Label('4. Point Inspector & MRV Metrics:', {fontWeight: 'bold', fontSize: '12px', margin: '12px 0 4px 0'});
panel.add(inspectorTitle);

var inspectorInfo = ui.Label('Click on any paddy pixel on the map to query pixel-level MRV metrics.', {fontSize: '11px', color: '#666'});
panel.add(inspectorInfo);

// Legend Widget
function createLegend(titleText, palette, minVal, maxVal, unit) {
  var legendPanel = ui.Panel({style: {margin: '8px 0 4px 0'}});
  var legendTitle = ui.Label(titleText + ' (' + unit + ')', {fontWeight: 'bold', fontSize: '11px'});
  legendPanel.add(legendTitle);

  var colorBar = ui.Thumbnail({
    image: ee.Image.pixelLonLat().select(0),
    params: {
      bbox: [0, 0, 1, 0.1],
      dimensions: '220x10',
      format: 'png',
      min: 0,
      max: 1,
      palette: palette
    },
    style: {stretch: 'horizontal', margin: '0 0 2px 0'}
  });
  legendPanel.add(colorBar);

  var labels = ui.Panel({
    widgets: [
      ui.Label(minVal.toString(), {fontSize: '10px'}),
      ui.Label(((minVal + maxVal) / 2).toString(), {fontSize: '10px', textAlign: 'center', stretch: 'horizontal'}),
      ui.Label(maxVal.toString(), {fontSize: '10px'})
    ],
    layout: ui.Panel.Layout.flow('horizontal'),
    style: {width: '220px'}
  });
  legendPanel.add(labels);
  return legendPanel;
}

panel.add(createLegend('Methane Flux', visFlux.palette, visFlux.min, visFlux.max, 'kg CH₄/ha'));
panel.add(createLegend('AWD Mitigation Potential', visAWD.palette, visAWD.min, visAWD.max, 'kg CH₄/ha'));

// Layer Update Function
function updateMapLayers() {
  map.layers().reset();

  var layers = getMethaneFluxLayer(currentYear);

  if (toggleS2.getValue()) {
    map.addLayer(layers.s2, visS2FCC, 'Sentinel-2 MSI FCC (' + currentYear + ')');
  }

  if (toggleSAR.getValue()) {
    map.addLayer(layers.s1, visSAR, 'Sentinel-1 SAR Composite (' + currentYear + ')');
  }

  if (toggleAWD.getValue()) {
    map.addLayer(layers.awd, visAWD, 'AWD Mitigation Potential (' + currentYear + ')', true, 0.85);
  }

  if (toggleMethane.getValue()) {
    map.addLayer(layers.flux, visFlux, 'IPCC Tier-2 Methane Flux (' + currentYear + ')', true, 0.85);
  }

  if (toggleDistricts.getValue()) {
    var districtOutline = ee.Image().byte().paint({
      featureCollection: cgDistricts,
      color: 1,
      width: 1.5
    });
    map.addLayer(districtOutline, {palette: ['#222222']}, '33 District Outlines');
  }
}

// ------------------------------------------------------------------------------
// 5. Interactive Pixel Click Listener
// ------------------------------------------------------------------------------
map.onClick(function(coords) {
  var point = ee.Geometry.Point([coords.lon, coords.lat]);
  var layers = getMethaneFluxLayer(currentYear);

  inspectorInfo.setValue('Querying pixel at Lat: ' + coords.lat.toFixed(4) + ', Lon: ' + coords.lon.toFixed(4) + '...');

  var sampleImage = layers.flux.rename('flux')
    .addBands(layers.awd.rename('awd'))
    .addBands(layers.s1.select('VV').rename('vv'))
    .addBands(layers.s2.select('NDVI').rename('ndvi'))
    .addBands(soilClay.rename('clay'))
    .addBands(soilSOC.rename('soc'));

  var sampleData = sampleImage.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: point,
    scale: 30
  });

  sampleData.evaluate(function(val) {
    if (!val || val.flux === null) {
      inspectorInfo.setValue('📍 Location: Non-rice pixel or upland terrain outside Kharif cultivation mask.');
      return;
    }

    var fluxKgHa = val.flux ? val.flux.toFixed(1) : 'N/A';
    var awdKgHa = val.awd ? val.awd.toFixed(1) : 'N/A';
    var vvDb = val.vv ? val.vv.toFixed(1) : 'N/A';
    var ndvi = val.ndvi ? val.ndvi.toFixed(2) : 'N/A';
    var clay = val.clay ? val.clay.toFixed(1) : 'N/A';
    var soc = val.soc ? val.soc.toFixed(1) : 'N/A';

    // Carbon offset calculation (AR6 GWP = 27.9)
    var co2eAvoided = (val.awd * 27.9 / 1000.0).toFixed(2);

    inspectorInfo.setValue(
      '📍 Selected Pixel MRV Attributes (' + currentYear + '):\n' +
      '• Methane Flux: ' + fluxKgHa + ' kg CH₄/ha/season\n' +
      '• AWD Mitigation: ' + awdKgHa + ' kg CH₄/ha (' + co2eAvoided + ' t CO₂e/ha)\n' +
      '• Sentinel-1 SAR VV: ' + vvDb + ' dB\n' +
      '• Sentinel-2 Peak NDVI: ' + ndvi + '\n' +
      '• SoilGrids Clay: ' + clay + '% | SOC: ' + soc + ' g/kg'
    );
  });
});

// Initial Render
updateMapLayers();
print('[INFO] Chhattisgarh Rice Methane MRV Dashboard loaded successfully.');
