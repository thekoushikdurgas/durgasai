from typing import List
import json

"""
Analyze a list of numbers with statistical operations.
"""

def number_list_analyzer(numbers: List[float], include_details: bool = None):
    """
    Analyze a list of numbers with statistical operations.
    
    Args:
        numbers: List of numbers to analyze (list of number items)
        include_details: Include detailed statistical analysis
    
    Returns:
        statistical_summary (object): Basic statistical measures (mean, median, min, max, etc.) (json format)
        analysis_report (string): Detailed statistical analysis report (plain_text format)
        dataset_size (integer): Number of data points analyzed
        outliers_detected (array[number]): List of detected outlier values (plain_text format)
    """
    try:
        # Handle empty dataset
        if not numbers:
            result = {
                "statistical_summary": {"error": "no_data_provided"},
                "analysis_report": "No numbers provided for analysis",
                "dataset_size": 0,
                "outliers_detected": []
            }
            return json.dumps(result, indent=2)
        
        # Basic calculations
        dataset_size = len(numbers)
        total = sum(numbers)
        mean = total / dataset_size
        sorted_nums = sorted(numbers)
        min_val = min(numbers)
        max_val = max(numbers)
        range_val = max_val - min_val
        
        # Calculate median
        if dataset_size % 2 == 1:
            median = sorted_nums[dataset_size // 2]
        else:
            median = (sorted_nums[dataset_size // 2 - 1] + sorted_nums[dataset_size // 2]) / 2
        
        # Calculate variance and standard deviation
        variance = sum((x - mean) ** 2 for x in numbers) / dataset_size
        std_dev = variance ** 0.5
        
        # Detect outliers using IQR method
        q1_idx = dataset_size // 4
        q3_idx = 3 * dataset_size // 4
        q1 = sorted_nums[q1_idx] if q1_idx < dataset_size else sorted_nums[0]
        q3 = sorted_nums[q3_idx] if q3_idx < dataset_size else sorted_nums[-1]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_detected = [x for x in numbers if x < lower_bound or x > upper_bound]
        
        # Create statistical summary object
        statistical_summary = {
            "count": dataset_size,
            "sum": round(total, 2),
            "mean": round(mean, 2),
            "median": round(median, 2),
            "min": min_val,
            "max": max_val,
            "range": round(range_val, 2),
            "variance": round(variance, 2),
            "standard_deviation": round(std_dev, 2),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "outlier_count": len(outliers_detected)
        }
        
        # Generate analysis report
        analysis_report = f"Statistical Analysis of {dataset_size} numbers:\n"
        analysis_report += f"Sum: {total}\n"
        analysis_report += f"Average: {mean:.2f}\n"
        analysis_report += f"Median: {median:.2f}\n"
        analysis_report += f"Min: {min_val}\n"
        analysis_report += f"Max: {max_val}\n"
        analysis_report += f"Range: {range_val}\n"
        analysis_report += f"Standard Deviation: {std_dev:.2f}\n"
        
        if include_details:
            analysis_report += f"\nDetailed Statistics:\n"
            analysis_report += f"Variance: {variance:.2f}\n"
            analysis_report += f"Q1 (25th percentile): {q1:.2f}\n"
            analysis_report += f"Q3 (75th percentile): {q3:.2f}\n"
            analysis_report += f"Outliers detected: {len(outliers_detected)}\n"
            if outliers_detected:
                analysis_report += f"Outlier values: {', '.join(map(str, outliers_detected))}\n"
            analysis_report += f"Dataset: {', '.join(map(str, numbers))}"
        
        # Return structured JSON output
        result = {
            "statistical_summary": statistical_summary,
            "analysis_report": analysis_report,
            "dataset_size": dataset_size,
            "outliers_detected": outliers_detected
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        # Return error in expected format
        error_result = {
            "statistical_summary": {"error": str(e)},
            "analysis_report": f"Error analyzing numbers: {str(e)}",
            "dataset_size": len(numbers) if numbers else 0,
            "outliers_detected": []
        }
        return json.dumps(error_result, indent=2)
