// main.js
document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const predictionForm = document.getElementById('prediction-form');
    const resultsContainer = document.getElementById('results-container');
    const loadingIndicator = document.getElementById('loading');
    const initialMessage = document.getElementById('initial-message');
    const predictedPriceElement = document.getElementById('predicted-price');
    const confidenceLevelElement = document.getElementById('confidence-level');
    const insightsElement = document.getElementById('insights');
    
    // Chart objects
    let priceChart = null;
    let importanceChart = null;
    
    // Event Listeners
    predictionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        makePrediction();
    });
    
    // Functions
    function makePrediction() {
        // Show loading indicator
        initialMessage.classList.add('d-none');
        resultsContainer.classList.add('d-none');
        loadingIndicator.classList.remove('d-none');
        
        // Get form data
        const formData = {
            crop_name: document.getElementById('crop_name').value,
            region: document.getElementById('region').value,
            season: document.getElementById('season').value,
            rainfall: parseFloat(document.getElementById('rainfall').value),
            soil_type: document.getElementById('soil_type').value,
            market_demand: document.getElementById('market_demand').value
        };
        
        // Send API request
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide loading, show results
            loadingIndicator.classList.add('d-none');
            resultsContainer.classList.remove('d-none');
            
            // Update prediction display
            predictedPriceElement.textContent = data.predicted_price;
            confidenceLevelElement.textContent = data.confidence_level;
            
            // Generate insights
            generateInsights(formData, data);
            
            // Get historical data and update charts
            fetchHistoricalData(formData.crop_name, formData.region)
                .then(historicalData => {
                    updatePriceChart(historicalData, data.predicted_price);
                    updateImportanceChart(data.feature_importance);
                });
        })
        .catch(error => {
            console.error('Error:', error);
            loadingIndicator.classList.add('d-none');
            initialMessage.classList.remove('d-none');
            alert('An error occurred during prediction. Please try again.');
        });
    }
    
    function fetchHistoricalData(crop, region) {
        return fetch(`/historical_data?crop=${crop}&region=${region}`)
            .then(response => response.json());
    }
    
    function updatePriceChart(historicalData, predictedPrice) {
        const ctx = document.getElementById('price-chart').getContext('2d');
        
        // Add predicted price to historical data
        const labels = [...historicalData.dates, 'Predicted'];
        const prices = [...historicalData.prices, predictedPrice];
        
        // Destroy previous chart if it exists
        if (priceChart) {
            priceChart.destroy();
        }
        
        // Create new chart
        priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Price (₹ per quintal)',
                    data: prices,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(75, 192, 192, 1)',
                    pointRadius: 4,
                    tension: 0.2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Price (₹)'
                        }
                    }
                }
            }
        });
    }
    
    function updateImportanceChart(featureImportance) {
        const ctx = document.getElementById('importance-chart').getContext('2d');
        
        // Convert feature importance to arrays
        const features = Object.keys(featureImportance);
        const values = Object.values(featureImportance);
        
        // Destroy previous chart if it exists
        if (importanceChart) {
            importanceChart.destroy();
        }
        
        // Create new chart
        importanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: features.map(f => f.charAt(0).toUpperCase() + f.slice(1).replace('_', ' ')),
                datasets: [{
                    label: 'Importance',
                    data: values,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.7)',
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(255, 206, 86, 0.7)',
                        'rgba(75, 192, 192, 0.7)',
                        'rgba(153, 102, 255, 0.7)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
    
    function generateInsights(formData, predictionData) {
        // Generate insights based on prediction and input data
        let insights = '';
        
        // Market demand impact
        if (formData.market_demand === 'High') {
            insights += '<p><strong>High Market Demand:</strong> Current high demand is positively influencing price.</p>';
        } else if (formData.market_demand === 'Low') {
            insights += '<p><strong>Low Market Demand:</strong> Weak demand is keeping prices down.</p>';
        }
        
        // Seasonal factors
        if (formData.season === 'Kharif') {
            insights += '<p><strong>Kharif Season:</strong> Monsoon crop prices tend to be lower due to higher production volume.</p>';
        } else if (formData.season === 'Rabi') {
            insights += '<p><strong>Rabi Season:</strong> Winter crops often command premium prices.</p>';
        }
        
        // Rainfall insights
        const rainfall = parseFloat(formData.rainfall);
        if (rainfall < 500) {
            insights += '<p><strong>Low Rainfall:</strong> Could affect yield and increase prices.</p>';
        } else if (rainfall > 1500) {
            insights += '<p><strong>Heavy Rainfall:</strong> May lead to oversupply in some crops.</p>';
        }
        
        // Add general recommendation
        insights += '<p><strong>Recommendation:</strong> Consider historical price trends before making planting or trading decisions.</p>';
        
        // Update insights element
        insightsElement.innerHTML = insights;
    }
});