import subprocess
import re
import json
#keeping this just in case 
# def calculate_score(error_count: int) -> float:
#     if error_count == 0:
#         return 5.0
#     elif error_count == 1:
#         return 4.5
#     elif error_count in [2, 3]:
#         return 4.0
#     elif error_count == 4:
#         return 3.5
#     elif error_count in [5, 6]:
#         return 3.0
#     elif error_count in [7, 8]:
#         return 2.5
#     elif error_count == 9:
#         return 2.0
#     elif error_count in [10, 11]:
#         return 1.5
#     elif error_count in [12, 13]:
#         return 1.0
#     elif error_count in [14, 15]:
#         return 0.5
#     else:
#         return 0.0

#to do later
def calculate_score(error_count: int) -> float:
    """Calculates score by deducting 0.75 marks for each error."""
    score = max(5.0 - (0.75 * error_count), 0.0)
    return round(score, 2)  # Rounds to 2 decimal places

def check_java_syntax(java_file_path: str) -> tuple[int, float, dict]:
    """
    Reads Java code from a file and checks for syntax errors using `javac`.
    Returns the error count, score, and a dictionary of errors.
    """
    error_dict = {}  # Dictionary to store errors by line number
    
    try:
        # Compile the Java file using javac
        result = subprocess.run(
            ["javac", java_file_path],
            capture_output=True,
            text=True
        )
        
        error_count = 0
        if result.stderr:
            print("Compilation Errors:")
            errors = result.stderr.strip().split('\n')
            
            # Process each error message
            for error in errors:
                # Skip the last line which contains error count
                if 'error' in error and not re.search(r'\d+ errors?$', error):
                    # Extract line number using regex
                    line_match = re.search(r':(\d+):', error)
                    if line_match:
                        line_num = int(line_match.group(1))
                        # Remove the file path and line number from the error message
                        error_msg = re.sub(r'^.*?:\d+:', '', error).strip()
                        
                        # Add error to dictionary
                        if line_num not in error_dict:
                            error_dict[line_num] = []
                        error_dict[line_num].append(error_msg)
                        print(f"- Line {line_num}: {error_msg}")

            last_line = errors[-1]
            if match := re.search(r'(\d+) errors?', last_line):
                error_count = int(match.group(1))
        else:
            print("No syntax errors found!")

        score = calculate_score(len(error_dict))
        print(f"\nCompiler reported errors: {error_count}")
        print(f"Number of lines with errors: {len(error_dict)}")
        print(f"Score: {score}/5")
        print("\nError Dictionary:")
        print(json.dumps(error_dict, indent=2))
        
        return error_count, score, error_dict
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1, 0.0, {}

java_file_path = "code.java"  #
error_count, score, error_dict = check_java_syntax(java_file_path)
