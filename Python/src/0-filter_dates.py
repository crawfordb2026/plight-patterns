#!/usr/bin/env python3
"""
Pipeline Step 0: Filter Monitor Files by Date Range

This script filters raw monitor files to extract specific date ranges.
It preserves the original tab-separated format and creates a new file with
a date-stamped filename.

Usage:
    python 0-filter_dates.py --input Monitor51 --load "06/20/25" --days 5 --offset 1
    
The script will automatically find the monitor file in Monitors_raw folder (e.g., Monitor51.txt).
Output is saved to monitors_date_filtered folder (Python/monitors_date_filtered/).
"""

import pandas as pd
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path


def parse_load_date(date_str):
    """
    Parse load date from MM/DD/YY format.
    
    Args:
        date_str: Date string in format "06/20/25" (MM/DD/YY)
        
    Returns:
        datetime object
    """
    try:
        # Parse MM/DD/YY format
        return pd.to_datetime(date_str, format='%m/%d/%y')
    except ValueError as e:
        raise ValueError(f"Invalid date format: {date_str}. Expected MM/DD/YY format (e.g., '06/20/25')") from e


def extract_monitor_number(monitor_name_or_path):
    """
    Extract monitor number from Monitor filename or path.
    
    Args:
        monitor_name_or_path: Monitor filename (e.g., "Monitor51" or "Monitor51.txt") or full path
        
    Returns:
        int: Monitor number
    """
    # Get just the filename from path if it's a full path
    filename = Path(monitor_name_or_path).stem
    # Remove .txt extension if present (though stem should already remove it)
    filename = filename.replace('.txt', '')
    # Remove "Monitor" or "monitor" prefix if present
    filename = filename.replace('Monitor', '').replace('monitor', '')
    # Extract digits from filename
    monitor_num = ''.join(filter(str.isdigit, filename))
    if not monitor_num:
        raise ValueError(f"Could not extract monitor number from: {monitor_name_or_path}. Expected format: Monitor51")
    return int(monitor_num)


def find_monitor_file(monitor_name):
    """
    Find monitor file in Monitors_raw folder.
    
    Args:
        monitor_name: Monitor filename (e.g., "Monitor51" or "Monitor51.txt")
        
    Returns:
        str: Full path to monitor file
    """
    # Remove .txt extension if present (we'll add it back)
    monitor_name_clean = monitor_name.replace('.txt', '')
    
    # Find Monitors_raw folder relative to script location
    script_dir = Path(__file__).parent
    # Go up to Python/, then to Monitors_raw
    monitors_raw_dir = script_dir.parent / 'Monitors_raw'
    
    # Construct filename
    filename = f"{monitor_name_clean}.txt"
    filepath = monitors_raw_dir / filename
    
    if filepath.exists():
        return str(filepath.absolute())
    
    # If not found, raise error
    raise FileNotFoundError(
        f"Monitor file not found: {filepath}. "
        f"Make sure the file exists in {monitors_raw_dir}"
    )


def filter_monitor_file(input_file, load_date_str, num_days, offset_days, output_file=None, monitor_name=None):
    """
    Filter monitor file by date range.
    
    Args:
        input_file: Path to input monitor file
        load_date_str: Load date in MM/DD/YY format (e.g., "06/20/25")
        num_days: Number of days of data to include after offset
        offset_days: Number of days to wait after load date before starting filter
        output_file: Optional output file path. If None, auto-generates from input filename.
        monitor_name: Original monitor name (e.g., "Monitor51") for output filename generation
        
    Returns:
        str: Path to output file
    """
    print("=" * 60)
    print("Filter Monitor File by Date Range")
    print("=" * 60)
    
    # Validate input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Parse load date
    load_date = parse_load_date(load_date_str)
    print(f"\nLoad date: {load_date.strftime('%B %d, %Y')} ({load_date_str})")
    print(f"Offset: {offset_days} day(s)")
    print(f"Days to include: {num_days} day(s)")
    
    # Calculate filter range: [load_date + offset @ 9am, load_date + offset + num_days @ 9am)
    filter_start = (load_date + timedelta(days=offset_days)).replace(hour=9, minute=0, second=0, microsecond=0)
    filter_end = (load_date + timedelta(days=offset_days + num_days)).replace(hour=9, minute=0, second=0, microsecond=0)
    
    print(f"\nFilter range: [{filter_start.strftime('%Y-%m-%d %H:%M:%S')}, {filter_end.strftime('%Y-%m-%d %H:%M:%S')})")
    
    # Generate output filename if not provided
    if output_file is None:
        # Use monitor_name if provided, otherwise extract from input_file
        if monitor_name:
            # Extract number from the original monitor name
            monitor_num = extract_monitor_number(monitor_name)
        else:
            # Extract from filename only (not full path)
            monitor_num = extract_monitor_number(Path(input_file).name)
        
        # Output to monitors_date_filtered folder
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / 'monitors_date_filtered'
        output_dir.mkdir(exist_ok=True)
        
        # Format dates for filename: Monitor{num}_{MM}_{DD}_{YY}.txt
        # Uses load date directly
        # Format: Monitor51_06_20_25.txt (MM, DD, YY from load date)
        # Example: load="06/20/25" → Monitor51_06_20_25.txt
        output_filename = f"Monitor{monitor_num}_{load_date.strftime('%m')}_{load_date.strftime('%d')}_{load_date.strftime('%y')}.txt"
        output_file = str(output_dir / output_filename)
    
    # Read file line by line and copy exact lines that fall within date range
    print(f"\nReading monitor file: {input_file}")
    output_file_abs = os.path.abspath(output_file)
    
    # Ensure output directory exists
    output_dir_path = os.path.dirname(output_file_abs)
    if output_dir_path and not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path, exist_ok=True)
    
    total_rows = 0
    filtered_rows = 0
    file_min_date = None
    file_max_date = None
    filtered_min = None
    filtered_max = None
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file_abs, 'w', encoding='utf-8', newline='') as outfile:
            
            for line in infile:
                total_rows += 1
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Parse the line - split by tab
                parts = line.rstrip('\n\r').split('\t')
                
                if len(parts) < 3:
                    # Skip lines that don't have at least date and time
                    continue
                
                # Extract date (column 1) and time (column 2)
                date_str = parts[1].strip()
                time_str = parts[2].strip()
                
                try:
                    # Parse datetime from monitor file format: "21 Feb 25 15:57:00"
                    line_datetime = pd.to_datetime(date_str + ' ' + time_str, format='%d %b %y %H:%M:%S')
                    
                    # Track file date range
                    if file_min_date is None or line_datetime < file_min_date:
                        file_min_date = line_datetime
                    if file_max_date is None or line_datetime > file_max_date:
                        file_max_date = line_datetime
                    
                    # Check if line is within filter range
                    if filter_start <= line_datetime < filter_end:
                        # Write the exact line as-is (preserve original line ending)
                        outfile.write(line)
                        filtered_rows += 1
                        
                        # Track filtered date range
                        if filtered_min is None or line_datetime < filtered_min:
                            filtered_min = line_datetime
                        if filtered_max is None or line_datetime > filtered_max:
                            filtered_max = line_datetime
                            
                except (ValueError, IndexError) as e:
                    # Skip lines that can't be parsed (but this shouldn't happen for valid monitor files)
                    continue
        
        print(f"   Total rows read: {total_rows:,}")
        if file_min_date and file_max_date:
            print(f"   Date range in file: {file_min_date.strftime('%Y-%m-%d %H:%M:%S')} to {file_max_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\nFiltered rows: {filtered_rows:,} (from {total_rows:,} total)")
        
        if filtered_rows == 0:
            print("WARNING: No data found in the specified date range!")
            # Remove empty output file
            if os.path.exists(output_file_abs):
                os.remove(output_file_abs)
            return None
        
        if filtered_min and filtered_max:
            print(f"   Filtered date range: {filtered_min.strftime('%Y-%m-%d %H:%M:%S')} to {filtered_max.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Verify file was created
        if os.path.exists(output_file_abs):
            file_size = os.path.getsize(output_file_abs)
            print(f"\n[OK] Successfully created filtered file: {output_file_abs}")
            print(f"   Rows written: {filtered_rows:,}")
            print(f"   File size: {file_size:,} bytes")
        else:
            raise IOError(f"File was not created at: {output_file_abs}")
            
    except Exception as e:
        raise IOError(f"Failed to process file: {e}") from e
    
    return output_file


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Filter monitor files by date range',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 0-filter_dates.py --input Monitor51 --load "06/20/25" --days 5 --offset 1
  python 0-filter_dates.py --input Monitor51 --load "06/20/25" --days 5 --offset 1 --output custom_output.txt
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Monitor filename (e.g., "Monitor51"). Will search in Monitors_raw folder.')
    parser.add_argument('--load', type=str, required=True,
                       help='Load date in MM/DD/YY format (e.g., "06/20/25"). This date will be used in the output filename.')
    parser.add_argument('--days', type=int, required=True,
                       help='Number of days of data to include after the offset (e.g., 5)')
    parser.add_argument('--offset', type=int, required=True,
                       help='Number of days to wait after load date before starting filter (e.g., 1 means filter starts 1 day after load date)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (optional, auto-generated and saved to Monitors_date_filtered folder)')
    
    args = parser.parse_args()
    
    try:
        # Find the monitor file
        input_file = find_monitor_file(args.input)
        print(f"Found monitor file: {input_file}")
        
        output_file = filter_monitor_file(
            input_file=input_file,
            load_date_str=args.load,
            num_days=args.days,
            offset_days=args.offset,
            output_file=args.output,
            monitor_name=args.input  # Pass original monitor name for output filename
        )
        
        if output_file:
            print("\n" + "=" * 60)
            print("Filtering complete!")
            print("=" * 60)
            sys.exit(0)
        else:
            print("\n" + "=" * 60)
            print("Filtering failed - no data in range")
            print("=" * 60)
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
