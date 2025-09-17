from typing import List
import json

"""
Manage a list of tasks with completion status.
"""

def task_list_manager(task_names: List[str], completion_status: List[bool], show_summary: bool = None):
    """
    Manage a list of tasks with completion status.
    
    Args:
        task_names: List of task names (list of string items)
        completion_status: List of completion status for each task (true/false) (list of boolean items)
        show_summary: Show task completion summary
    
    Returns:
        task_report (string): Formatted task status report (plain_text format)
        completion_summary (object): Summary statistics of task completion (json format)
        completed_tasks (array[string]): List of completed task names (plain_text format)
        pending_tasks (array[string]): List of pending task names (plain_text format)
    """
    try:
        # Validate that both lists have the same length
        if len(task_names) != len(completion_status):
            error_result = {
                "task_report": "Error: task_names and completion_status lists must have the same length",
                "completion_summary": {"error": "mismatched_list_lengths"},
                "completed_tasks": [],
                "pending_tasks": []
            }
            return json.dumps(error_result, indent=2)
        
        # Initialize lists for tracking completed and pending tasks
        completed_tasks = []
        pending_tasks = []
        
        # Process tasks
        task_report = ""
        if task_names:
            task_report += "Task List Status:\n"
            task_report += "=" * 20 + "\n"
            
            for i, (task, is_completed) in enumerate(zip(task_names, completion_status), 1):
                status_icon = "✅" if is_completed else "❌"
                status_text = "COMPLETED" if is_completed else "PENDING"
                task_report += f"{i}. {status_icon} {task.strip()} [{status_text}]\n"
                
                if is_completed:
                    completed_tasks.append(task.strip())
                else:
                    pending_tasks.append(task.strip())
            
            # Add summary to report if requested
            if show_summary:
                task_report += "\n" + "="*20 + "\n"
                task_report += "SUMMARY:\n"
                task_report += f"Total Tasks: {len(task_names)}\n"
                task_report += f"Completed: {len(completed_tasks)} ({len(completed_tasks)/len(task_names)*100:.1f}%)\n"
                task_report += f"Pending: {len(pending_tasks)} ({len(pending_tasks)/len(task_names)*100:.1f}%)\n"
                
                if completed_tasks:
                    task_report += f"\nCompleted Tasks:\n• " + "\n• ".join(completed_tasks)
                if pending_tasks:
                    task_report += f"\n\nPending Tasks:\n• " + "\n• ".join(pending_tasks)
        else:
            task_report = "No tasks provided for management"
        
        # Create completion summary object
        completion_summary = {
            "total_tasks": len(task_names),
            "completed_count": len(completed_tasks),
            "pending_count": len(pending_tasks),
            "completion_percentage": round((len(completed_tasks) / len(task_names) * 100), 1) if task_names else 0,
            "status": "success"
        }
        
        # Return structured JSON output
        result = {
            "task_report": task_report,
            "completion_summary": completion_summary,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        # Return error in expected format
        error_result = {
            "task_report": f"Error managing task list: {str(e)}",
            "completion_summary": {"error": str(e), "status": "failed"},
            "completed_tasks": [],
            "pending_tasks": []
        }
        return json.dumps(error_result, indent=2)
