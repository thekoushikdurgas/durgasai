from typing import List
import json

"""
Advanced data analysis tool demonstrating input/output parameter structure.
"""

def data_analyzer_enhanced(dataset: List[float], analysis_type: str, include_visualization: bool = None):
    """
    Advanced data analysis tool demonstrating input/output parameter structure.
    
    Args:
        dataset: Array of numerical data points to analyze (list of number items)
        analysis_type: Type of analysis: basic, detailed, or statistical
        include_visualization: Whether to include visualization data
    
    Returns:
        summary (object): Statistical summary of the dataset (json format)
        analysis_report (string): Detailed analysis report (markdown format)
        data_quality_score (number): Quality score of the input data (0-100)
        recommendations (array[string]): List of recommendations based on analysis (plain_text format)
    """
    try:
        if not dataset:
            return json.dumps({
                "error": "No data provided",
                "summary": {},
                "analysis_report": "# Error\nNo data points provided for analysis.",
                "data_quality_score": 0,
                "recommendations": ["Provide valid numerical data for analysis"]
            })
        
        # Calculate basic statistics
        total = sum(dataset)
        count = len(dataset)
        mean = total / count
        sorted_data = sorted(dataset)
        median = sorted_data[count//2] if count % 2 == 1 else (sorted_data[count//2-1] + sorted_data[count//2]) / 2
        min_val = min(dataset)
        max_val = max(dataset)
        range_val = max_val - min_val
        
        # Calculate variance and standard deviation
        variance = sum((x - mean) ** 2 for x in dataset) / count
        std_dev = variance ** 0.5
        
        # Data quality assessment
        outlier_threshold = 2 * std_dev
        outliers = [x for x in dataset if abs(x - mean) > outlier_threshold]
        data_quality_score = max(0, 100 - (len(outliers) / count * 100))
        
        # Generate summary object
        summary = {
            "count": count,
            "mean": round(mean, 2),
            "median": round(median, 2),
            "min": min_val,
            "max": max_val,
            "range": range_val,
            "variance": round(variance, 2),
            "std_deviation": round(std_dev, 2),
            "outliers_count": len(outliers)
        }
        
        # Generate analysis report based on type
        if analysis_type.lower() == "basic":
            analysis_report = f"""# Basic Data Analysis Report
            
## Dataset Overview
- **Count**: {count} data points
- **Mean**: {mean:.2f}
- **Median**: {median:.2f}
- **Range**: {min_val} to {max_val}

## Summary
The dataset contains {count} numerical values with an average of {mean:.2f}.
"""
        elif analysis_type.lower() == "detailed":
            analysis_report = f"""# Detailed Data Analysis Report
            
## Dataset Overview
- **Count**: {count} data points
- **Mean**: {mean:.2f}
- **Median**: {median:.2f}
- **Min/Max**: {min_val} / {max_val}
- **Range**: {range_val}
- **Standard Deviation**: {std_dev:.2f}
- **Variance**: {variance:.2f}

## Data Distribution
- **Outliers Detected**: {len(outliers)} ({len(outliers)/count*100:.1f}% of data)
- **Data Quality Score**: {data_quality_score:.1f}/100

## Analysis
The dataset shows {'high' if std_dev/mean < 0.1 else 'moderate' if std_dev/mean < 0.3 else 'high'} variability 
with a coefficient of variation of {(std_dev/mean)*100:.1f}%.
"""
        else:  # statistical
            skewness = "right" if mean > median else "left" if mean < median else "symmetric"
            analysis_report = f"""# Statistical Analysis Report
            
## Descriptive Statistics
| Metric | Value |
|--------|--------|
| Count | {count} |
| Mean | {mean:.2f} |
| Median | {median:.2f} |
| Mode | {max(set(dataset), key=dataset.count)} |
| Standard Deviation | {std_dev:.2f} |
| Variance | {variance:.2f} |
| Coefficient of Variation | {(std_dev/mean)*100:.1f}% |

## Distribution Analysis
- **Skewness**: {skewness} skewed
- **Outliers**: {len(outliers)} detected
- **Data Quality**: {data_quality_score:.1f}/100

## Statistical Insights
The distribution appears to be {skewness} skewed with {'low' if len(outliers) < count*0.05 else 'moderate' if len(outliers) < count*0.1 else 'high'} 
outlier presence affecting data quality.
"""
        
        # Generate recommendations
        recommendations = []
        
        if len(outliers) > count * 0.1:
            recommendations.append("Consider investigating outliers - they may indicate data quality issues")
        
        if std_dev / mean > 0.5:
            recommendations.append("High variability detected - consider data normalization")
        
        if count < 30:
            recommendations.append("Sample size is small - consider collecting more data for robust analysis")
        
        if data_quality_score < 70:
            recommendations.append("Data quality is below recommended threshold - review data collection process")
        
        if not recommendations:
            recommendations.append("Data looks good for further analysis")
        
        # Add visualization recommendation if requested
        if include_visualization:
            recommendations.append("Consider creating histogram and box plot visualizations for better insights")
        
        # Return structured output (in real implementation, this would return the actual structured data)
        # For demonstration, we'll return a JSON string containing all outputs
        result = {
            "summary": summary,
            "analysis_report": analysis_report,
            "data_quality_score": round(data_quality_score, 1),
            "recommendations": recommendations
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": f"Analysis failed: {str(e)}",
            "summary": {},
            "analysis_report": "# Error\nAnalysis could not be completed.",
            "data_quality_score": 0,
            "recommendations": ["Check input data format and try again"]
        })
