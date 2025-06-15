// WebSocket connection
const socket = io();

// Chart setup
const ctx = document.getElementById('equityChart').getContext('2d');
const equityChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Portfolio Value',
            data: [],
            borderColor: '#3fb950',
            tension: 0.1
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            y: {
                beginAtZero: false,
                grid: {
                    color: '#30363d'
                }
            },
            x: {
                grid: {
                    color: '#30363d'
                }
            }
        }
    }
});

// Socket event handlers
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('status_update', (data) => {
    updateDashboard(data);
});

socket.on('trades_update', (trades) => {
    updateTradesTable(trades);
});

socket.on('positions_update', (positions) => {
    updatePositions(positions);
});

// UI functions
function updateDashboard(data) {
    document.getElementById('portfolioValue').textContent = `$${data.current_equity.toFixed(2)}`;
    document.getElementById('dailyPnl').textContent = `$${data.daily_pnl.toFixed(2)}`;
    document.getElementById('openPositions').textContent = Object.keys(data.positions).length;
    
    if (data.daily_pnl >= 0) {
        document.getElementById('dailyPnl').className = 'text-success';
    } else {
        document.getElementById('dailyPnl').className = 'text-danger';
    }
}

// Start/Stop trading buttons
document.getElementById('startTradingBtn').addEventListener('click', () => {
    const symbols = prompt('Enter symbols to trade (comma-separated):', 'AAPL,GOOGL,MSFT');
    if (symbols) {
        socket.emit('start_trading', {
            symbols: symbols.split(',').map(s => s.trim()),
            paper_trading: true
        });
        
        document.getElementById('startTradingBtn').style.display = 'none';
        document.getElementById('stopTradingBtn').style.display = 'inline-block';
    }
});

document.getElementById('stopTradingBtn').addEventListener('click', () => {
    socket.emit('stop_trading');
    document.getElementById('startTradingBtn').style.display = 'inline-block';
    document.getElementById('stopTradingBtn').style.display = 'none';
});