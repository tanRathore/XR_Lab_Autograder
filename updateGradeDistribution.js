// Modified updateGradeDistribution function that can handle different data sources
window.updateGradeDistribution = function(type, specificStudentScores = null, specificMaxScore = null) {
    const ctx = document.getElementById('gradeDistribution').getContext('2d');
    let labels, distributionData, originalDistributionData, chartType = 'bar', chartLabel = 'Current Distribution';

    // Use provided scores or default to window values
    const studentScores = specificStudentScores || window.currentStudentScores;
    const maxPossibleScore = specificMaxScore || window.currentMaxPossibleScore;
    
    if (!studentScores || Object.keys(studentScores).length === 0) return;
    
    // Calculate grades as percentages
    const totalGrades = Object.values(studentScores);
    const percentages = totalGrades.map(grade => (grade / maxPossibleScore) * 100);
    
    // Check if we have original data and if it's different from current data
    const hasOriginalData = originalStudentScores && 
                           JSON.stringify(studentScores) !== JSON.stringify(originalStudentScores);
    
    // Calculate original percentages if available
    let originalPercentages = [];
    if (hasOriginalData) {
        const originalTotalGrades = Object.values(originalStudentScores);
        originalPercentages = originalTotalGrades.map(grade => 
            (grade / originalMaxPossibleScore) * 100);
    }

    if (type === 'curve') {
        // Simple curve: histogram as a line chart
        chartType = 'line';

        // Create bins (e.g., 20 bins from 0 to 100)
        const binCount = 20;
        const binSize = 100 / binCount;
        labels = [];
        distributionData = new Array(binCount).fill(0);
        originalDistributionData = new Array(binCount).fill(0);

        for (let i = 0; i < binCount; i++) {
            labels.push(`${Math.round(i * binSize)}-${Math.round((i + 1) * binSize)}`);
        }

        percentages.forEach(percentage => {
            let bin = Math.floor(percentage / binSize);
            if (bin >= binCount) bin = binCount - 1;
            distributionData[bin]++;
        });
        
        // Fill original data bins if available
        if (hasOriginalData) {
            originalPercentages.forEach(percentage => {
                let bin = Math.floor(percentage / binSize);
                if (bin >= binCount) bin = binCount - 1;
                originalDistributionData[bin]++;
            });
        }
    } else {
        // Original bar chart logic
        let ranges, labelsArr;
        if (type === 'fixed') {
            ranges = [0, 60, 70, 80, 90, 100];
            labelsArr = ['0-59', '60-69', '70-79', '80-89', '90-100'];
        } else {
            const minGrade = Math.min(...percentages);
            const maxGrade = Math.max(...percentages);
            const range = maxGrade - minGrade;
            const binSize = range / 5;
            ranges = Array.from({length: 6}, (_, i) => minGrade + (binSize * i));
            labelsArr = ranges.slice(0, -1).map((val, i) => `${val.toFixed(1)}-${ranges[i + 1].toFixed(1)}`);
        }
        labels = labelsArr;
        distributionData = new Array(ranges.length - 1).fill(0);
        originalDistributionData = new Array(ranges.length - 1).fill(0);
        
        percentages.forEach(percentage => {
            for (let i = 0; i < ranges.length - 1; i++) {
                if (percentage >= ranges[i] && percentage <= ranges[i + 1]) {
                    distributionData[i]++;
                    break;
                }
            }
        });
        
        // Fill original data bins if available
        if (hasOriginalData) {
            originalPercentages.forEach(percentage => {
                for (let i = 0; i < ranges.length - 1; i++) {
                    if (percentage >= ranges[i] && percentage <= ranges[i + 1]) {
                        originalDistributionData[i]++;
                        break;
                    }
                }
            });
        }
    }

    // Update the chart
    if (window.gradeChart) {
        window.gradeChart.destroy();
    }
    
    // Prepare datasets array - always include current distribution
    const datasets = [{
        label: 'Current Distribution',
        data: distributionData,
        backgroundColor: chartType === 'bar' ? 'rgba(54, 162, 235, 0.5)' : 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 2,
        fill: false,
        tension: 0.3 // smooth curve
    }];
    
    // Add original distribution if different from current
    if (hasOriginalData) {
        datasets.push({
            label: 'Original Distribution',
            data: originalDistributionData,
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 2,
            borderDash: [5, 5], // dashed line
            fill: false,
            tension: 0.3
        });
    }
    
    window.gradeChart = new Chart(ctx, {
        type: chartType,
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Students'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Percentage'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: type === 'curve' ? 'Grade Distribution (Curve)' : 'Grade Distribution'
                },
                legend: {
                    display: hasOriginalData, // only show legend if we have two datasets
                    position: 'top'
                }
            }
        }
    });
}; 