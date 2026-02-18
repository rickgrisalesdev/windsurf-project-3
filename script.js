class TemperatureConverter {
    constructor() {
        this.temperatureInput = document.getElementById('temperature');
        this.unitRadios = document.querySelectorAll('input[name="unit"]');
        this.celsiusResult = document.getElementById('celsius-result');
        this.fahrenheitResult = document.getElementById('fahrenheit-result');
        this.kelvinResult = document.getElementById('kelvin-result');
        this.quickButtons = document.querySelectorAll('.quick-btn');
        
        this.init();
    }

    init() {
        // Event listeners for real-time conversion
        this.temperatureInput.addEventListener('input', () => this.convert());
        this.unitRadios.forEach(radio => {
            radio.addEventListener('change', () => this.convert());
        });

        // Event listeners for quick conversion buttons
        this.quickButtons.forEach(button => {
            button.addEventListener('click', () => {
                const value = parseFloat(button.dataset.value);
                const unit = button.dataset.unit;
                
                this.temperatureInput.value = value;
                document.querySelector(`input[name="unit"][value="${unit}"]`).checked = true;
                this.convert();
                
                // Add visual feedback
                button.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    button.style.transform = '';
                }, 100);
            });
        });

        // Initial conversion
        this.convert();
    }

    convert() {
        const value = parseFloat(this.temperatureInput.value);
        
        if (isNaN(value)) {
            this.clearResults();
            return;
        }

        const selectedUnit = document.querySelector('input[name="unit"]:checked').value;
        let celsius, fahrenheit, kelvin;

        // Convert to Celsius first
        switch (selectedUnit) {
            case 'celsius':
                celsius = value;
                break;
            case 'fahrenheit':
                celsius = this.fahrenheitToCelsius(value);
                break;
            case 'kelvin':
                celsius = this.kelvinToCelsius(value);
                break;
        }

        // Convert from Celsius to other units
        fahrenheit = this.celsiusToFahrenheit(celsius);
        kelvin = this.celsiusToKelvin(celsius);

        // Display results with formatting
        this.displayResults(celsius, fahrenheit, kelvin);
    }

    celsiusToFahrenheit(celsius) {
        return (celsius * 9/5) + 32;
    }

    fahrenheitToCelsius(fahrenheit) {
        return (fahrenheit - 32) * 5/9;
    }

    celsiusToKelvin(celsius) {
        return celsius + 273.15;
    }

    kelvinToCelsius(kelvin) {
        return kelvin - 273.15;
    }

    displayResults(celsius, fahrenheit, kelvin) {
        this.celsiusResult.textContent = this.formatNumber(celsius);
        this.fahrenheitResult.textContent = this.formatNumber(fahrenheit);
        this.kelvinResult.textContent = this.formatNumber(kelvin);

        // Add color coding based on temperature ranges
        this.addTemperatureColoring(this.celsiusResult, celsius, 'celsius');
        this.addTemperatureColoring(this.fahrenheitResult, fahrenheit, 'fahrenheit');
        this.addTemperatureColoring(this.kelvinResult, kelvin, 'kelvin');
    }

    formatNumber(num) {
        if (Math.abs(num) < 0.01 && Math.abs(num) > 0) {
            return num.toExponential(2);
        }
        return num.toFixed(2);
    }

    addTemperatureColoring(element, temp, unit) {
        element.style.color = '';
        
        let celsiusTemp = temp;
        if (unit === 'fahrenheit') {
            celsiusTemp = this.fahrenheitToCelsius(temp);
        } else if (unit === 'kelvin') {
            celsiusTemp = this.kelvinToCelsius(temp);
        }

        // Color coding based on Celsius temperature
        if (celsiusTemp <= -50) {
            element.style.color = '#0066cc'; // Very cold - deep blue
        } else if (celsiusTemp <= 0) {
            element.style.color = '#3399ff'; // Freezing - light blue
        } else if (celsiusTemp <= 15) {
            element.style.color = '#66ccff'; // Cold - very light blue
        } else if (celsiusTemp <= 25) {
            element.style.color = '#667eea'; // Normal - default purple
        } else if (celsiusTemp <= 35) {
            element.style.color = '#ff9933'; // Warm - orange
        } else if (celsiusTemp <= 50) {
            element.style.color = '#ff6600'; // Hot - dark orange
        } else {
            element.style.color = '#cc0000'; // Very hot - red
        }
    }

    clearResults() {
        this.celsiusResult.textContent = '--';
        this.fahrenheitResult.textContent = '--';
        this.kelvinResult.textContent = '--';
        
        // Reset colors
        this.celsiusResult.style.color = '';
        this.fahrenheitResult.style.color = '';
        this.kelvinResult.style.color = '';
    }

    // Utility method for batch conversion
    convertMultiple(values, fromUnit) {
        return values.map(value => {
            let celsius;
            switch (fromUnit) {
                case 'celsius':
                    celsius = value;
                    break;
                case 'fahrenheit':
                    celsius = this.fahrenheitToCelsius(value);
                    break;
                case 'kelvin':
                    celsius = this.kelvinToCelsius(value);
                    break;
            }
            
            return {
                celsius: celsius,
                fahrenheit: this.celsiusToFahrenheit(celsius),
                kelvin: this.celsiusToKelvin(celsius)
            };
        });
    }

    // Method to get temperature description
    getTemperatureDescription(celsius) {
        if (celsius <= -273.15) return 'Absoluto cero (imposible)';
        if (celsius <= -50) return 'Extremadamente frío';
        if (celsius <= -20) return 'Muy frío';
        if (celsius <= 0) return 'Congelando';
        if (celsius <= 10) return 'Frío';
        if (celsius <= 20) return 'Fresco';
        if (celsius <= 25) return 'Agradable';
        if (celsius <= 30) return 'Cálido';
        if (celsius <= 35) return 'Caluroso';
        if (celsius <= 40) return 'Muy caluroso';
        return 'Extremadamente caluroso';
    }
}

// Initialize the converter when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const converter = new TemperatureConverter();
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            switch(e.key) {
                case '1':
                    e.preventDefault();
                    document.querySelector('input[name="unit"][value="celsius"]').checked = true;
                    converter.convert();
                    break;
                case '2':
                    e.preventDefault();
                    document.querySelector('input[name="unit"][value="fahrenheit"]').checked = true;
                    converter.convert();
                    break;
                case '3':
                    e.preventDefault();
                    document.querySelector('input[name="unit"][value="kelvin"]').checked = true;
                    converter.convert();
                    break;
                case 'l':
                    e.preventDefault();
                    document.getElementById('temperature').focus();
                    break;
            }
        }
        
        // Clear input on Escape
        if (e.key === 'Escape') {
            document.getElementById('temperature').value = '';
            converter.clearResults();
        }
    });

    // Add smooth scroll behavior
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add input validation
    const temperatureInput = document.getElementById('temperature');
    temperatureInput.addEventListener('keypress', (e) => {
        // Allow only numbers, decimal point, minus sign, and control keys
        const char = String.fromCharCode(e.which);
        if (!/[0-9.\-]/.test(char) && e.which !== 8 && e.which !== 46 && e.which !== 45) {
            e.preventDefault();
        }
    });

    console.log('🌡️ Conversor de Temperatura initialized successfully!');
    console.log('Keyboard shortcuts:');
    console.log('Ctrl/Cmd + 1: Select Celsius');
    console.log('Ctrl/Cmd + 2: Select Fahrenheit');
    console.log('Ctrl/Cmd + 3: Select Kelvin');
    console.log('Ctrl/Cmd + L: Focus input');
    console.log('Escape: Clear input');
});
