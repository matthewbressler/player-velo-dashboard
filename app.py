from flask import Flask, render_template, request
from flask_caching import Cache
from collections import defaultdict
from datetime import datetime
import re
import os
import google.auth
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI plotting
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from googleapiclient.discovery import build
from google.oauth2 import service_account


# Path to the credentials JSON file
SERVICE_ACCOUNT_FILE = 'credentials/player-velo-dashboard-3934aa36cdba.json'

# Scopes required by the API
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

# ID of the Google Sheet
SPREADSHEET_ID = '1AkHG-KgPgZrCKpoCRk-M80UWXmQm9RMGOhwyT0CMKTk'

# Range of the data to fetch from the sheet
RANGE_NAME = 'Form Responses 1!A:K'  # Change the range as needed


app = Flask(__name__)
# Configure cache (SimpleCache is an in-memory cache for development/testing)
app.config['CACHE_TYPE'] = 'SimpleCache'  

app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # Cache timeout in seconds (adjust as needed)

# Initialize the cache
cache = Cache(app)




# Create Sheets API Client
# Cache the Google Sheets data
@cache.cached(timeout=600, key_prefix='sheet_data')
def get_google_sheet_data():
    # Load credentials from the service account file
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    
    # Build the Sheets API client
    service = build('sheets', 'v4', credentials=creds)
    
    # Fetch the data from the sheet
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
    
    # Get the values from the response
    values = result.get('values', [])
    
    return values



# Cache player-specific report generation
@cache.cached(timeout=300, key_prefix=lambda: f"player_report_{request.args.get('player_name', 'default')}")
def generate_player_report(player_name):
    """
    Generates the session metrics, heatmap and line graph for a specific player.
    The results are cached for 5 minutes (300 seconds).
    """
    # Fetch data from the sheet (cached)
    sheet_data = get_google_sheet_data()

    # Process player data and calculate session metrics
    player_sessions = process_player_data(sheet_data)
    session_metrics = calculate_session_metrics(player_sessions)

    # Check if player exists in data
    if player_name not in session_metrics:
        return None

    player_data = session_metrics[player_name]
    df = build_player_dataframe(player_data)

    # Generate heatmap and save it to a file
    heatmap_image_path = f'static/heatmap_{player_name.replace(" ", "_")}.png'
    plot_session_performance_heatmap(df, heatmap_image_path)

    # Generate line graph of max velocity and save it to a file
    line_graph_image_path = f'static/line_graph_{player_name.replace(" ", "_")}.png'
    plot_max_velocity_line_graph(df, line_graph_image_path)

    # Prepare scatter data for the player
    scatter_data = prepare_scatter_data({player_name: player_data})

    # Generate interactive scatter plot HTML
    scatter_plot_html = plot_scatter_interactive(scatter_data)

    return {
        'player_data': player_data,
        'player_profile': {
            'player_name' : player_name,
            'age' : player_sessions[player_name][0]['age'],
            'email': player_sessions[player_name][0]['email'],
            'all_time_max_velocity': player_data['all_time_max_velocity']
        },
        'heatmap_image_path': heatmap_image_path,
        'line_graph_image_path': line_graph_image_path,  # Add the path for the line graph
        'scatter_plot_html': scatter_plot_html
    }

def get_unique_player_names(sheet_data):
    # Extract the "Player Name" column
    player_names = [row[2] for row in sheet_data[1:] if len(row) > 2]

    # Return sorted list of tuples (normalized_name, url_friendly_name)
    return sorted(set(player_names))  # Sort by normalized name



def filter_player_data(sheet_data, player_name):
    # Filter the sheet data to include only the rows that match the player name
    return [row for row in sheet_data if row[2].lower() == player_name.lower()]



def process_player_data(sheet_data):
    """
    Processes sheet data for each player.

    Args:
    sheet_data (list of lists)

    Returns:
        dafaultdict: dictionary where keys are player names, and values are lists of dictionaries containing
        information per session
    
    """
    # Initialize a defaultdict to store velocities by session for each player
    player_sessions = defaultdict(list)

    for row in sheet_data[1:]:  # Skip header row
        player_name = row[2]  # Player name is in the third column
        email = row[1]  # Email is in the second column
        session_date = row[3]
        velocity_100_str = row[4]  # 100% velocities in the fifth column
        velocity_80_str = row[6]  # 80% velocities in the seventh column
        velocity_90_str = row[7]  # 90% velocities in the eighth column
        age = row[8]
        problems = row[9]
        arm_feel = row[10]


        # Process raw velocities and combine them into a single list
        combined_velocities = []

        # Store the velocities for each intent (separated)
        velocities_100 = []
        velocities_80 = []
        velocities_90 = []

        personal_note = row[5] if len(row) > 5 else ""
        if not personal_note:  # If personal note is empty, set it to "No notes"
            personal_note = "No notes"

        ## Add the velocities from each intent to the corresponding lists
        for velocity_str, intent_list in zip([velocity_80_str, velocity_90_str, velocity_100_str],
                                             [velocities_80, velocities_90, velocities_100]):
            if velocity_str:  # Only process if the velocity string is not empty
                # Extract and filter velocities, ensuring all entries are integers
                for v in velocity_str.replace(" ", ",").split(','):
                    v = v.strip()  # Clean up the value
                    if v.isdigit():  # Only add numbers that are valid integers
                        intent_list.append(int(v))
                        combined_velocities.append(int(v))  # Add to combined list as well

            
        # Append the session data (separated velocities and combined velocities) to the player's sessions
        player_sessions[player_name].append({
            'email': email,  # Store player email
            'age' : age,
            'date': session_date,  # Store session date as is
            'velocities_100': velocities_100,  # Store 100% velocities
            'velocities_80': velocities_80,  # Store 80% velocities
            'velocities_90': velocities_90,  # Store 90% velocities
            'combined_velocities': combined_velocities,  # Store combined velocities
            'personal_note': personal_note,  # Store personal note
            'problems' : problems, # Store problems from session
            'arm_feel' : arm_feel # Store how the arms felt during session
        })

    return player_sessions


def calculate_session_metrics(player_sessions):
    player_summary = {}

    for player_name, sessions in player_sessions.items():
        # Sort sessions by date in descending order
        sessions.sort(key=lambda x: datetime.strptime(x['date'], "%m/%d/%Y"), reverse=True)

        session_metrics = []  # Stores metrics for each session for the current player
        all_time_max_velocity = 0  # Variable to track the all-time max velocity

        for session in sessions:
            velocities_100 = session['velocities_100']
            velocities_80 = session['velocities_80']
            velocities_90 = session['velocities_90']
            combined_velocities = session['combined_velocities']  # New key for combined velocities
            session_date = session['date']
            personal_note = session['personal_note']
            problems = session['problems']
            arm_feel = session['arm_feel']

            # Update all-time max velocity (including combined velocities)
            all_time_max_velocity = max(all_time_max_velocity, max(combined_velocities) if combined_velocities else 0)


            # Calculate metrics for 100% intent velocities
            if velocities_100:
                avg_velocity_100 = round(sum(velocities_100) / len(velocities_100), 2)
                max_velocity_100 = max(velocities_100)
                min_velocity_100 = min(velocities_100)
                velocity_range_100 = max_velocity_100 - min_velocity_100
            else:
                avg_velocity_100 = max_velocity_100 = min_velocity_100 = velocity_range_100 = 0

            # Calculate metrics for 80% intent velocities
            if velocities_80:
                avg_velocity_80 = round(sum(velocities_80) / len(velocities_80), 2)
                max_velocity_80 = max(velocities_80)
                min_velocity_80 = min(velocities_80)
                velocity_range_80 = max_velocity_80 - min_velocity_80
            else:
                avg_velocity_80 = max_velocity_80 = min_velocity_80 = velocity_range_80 = 0

            # Calculate metrics for 90% intent velocities
            if velocities_90:
                avg_velocity_90 = round(sum(velocities_90) / len(velocities_90), 2)
                max_velocity_90 = max(velocities_90)
                min_velocity_90 = min(velocities_90)
                velocity_range_90 = max_velocity_90 - min_velocity_90
            else:
                avg_velocity_90 = max_velocity_90 = min_velocity_90 = velocity_range_90 = 0

            # Calculate metrics for combined velocities
            if combined_velocities:
                avg_combined_velocity = round(sum(combined_velocities) / len(combined_velocities), 2)
                max_combined_velocity = max(combined_velocities)
                min_combined_velocity = min(combined_velocities)
                velocity_range_combined = max_combined_velocity - min_combined_velocity
            else:
                avg_combined_velocity = max_combined_velocity = min_combined_velocity = velocity_range_combined = 0



            # Append session-level metrics for all intents and combined velocities
            session_metrics.append({
                'date': session_date,
                'velocities_100': velocities_100,
                'avg_velocity_100': avg_velocity_100,
                'max_velocity_100': max_velocity_100,
                'min_velocity_100': min_velocity_100,
                'velocity_range_100': velocity_range_100,
                'velocities_80': velocities_80,
                'avg_velocity_80': avg_velocity_80,
                'max_velocity_80': max_velocity_80,
                'min_velocity_80': min_velocity_80,
                'velocity_range_80': velocity_range_80,
                'velocities_90': velocities_90,
                'avg_velocity_90': avg_velocity_90,
                'max_velocity_90': max_velocity_90,
                'min_velocity_90': min_velocity_90,
                'velocity_range_90': velocity_range_90,
                'combined_velocities': combined_velocities,
                'avg_combined_velocity': avg_combined_velocity,  # New metric for combined velocities
                'max_combined_velocity': max_combined_velocity,  # New metric for combined velocities
                'min_combined_velocity': min_combined_velocity,  # New metric for combined velocities
                'velocity_range_combined': velocity_range_combined,  # New metric for combined velocities
                'personal_note': personal_note,
                'problems' : problems,
                'arm_feel' : arm_feel
            })

        # Store the session metrics and all-time max velocity for the current player
        player_summary[player_name] = {
            'session_metrics': session_metrics,
            'all_time_max_velocity': all_time_max_velocity
        }

        

    return player_summary



def build_player_dataframe(player_data):
    """
    Converts the session metrics into a DataFrame for visualization.

    Args:
    player_data (dict): Dictionary containing player session data.

    Returns:
    pd.DataFrame: DataFrame with columns for date, velocity, avg_velocity, etc.
    """
    
    data = []
    
    for session in player_data['session_metrics']:
        # Ensure the date is treated as a string (no need for integer conversion)
        session_date = session['date']  # This is assumed to be a string like 'mm/dd/yyyy'
        avg_velocity_100 = session['avg_velocity_100']
        max_velocity_100 = session['max_velocity_100']
        min_velocity_100 = session['min_velocity_100']
        velocity_range_100 = session['velocity_range_100']
        
        avg_velocity_80 = session['avg_velocity_80']
        max_velocity_80 = session['max_velocity_80']
        min_velocity_80 = session['min_velocity_80']
        velocity_range_80 = session['velocity_range_80']
        
        avg_velocity_90 = session['avg_velocity_90']
        max_velocity_90 = session['max_velocity_90']
        min_velocity_90 = session['min_velocity_90']
        velocity_range_90 = session['velocity_range_90']
        
        avg_combined_velocity = session['avg_combined_velocity']
        max_combined_velocity = session['max_combined_velocity']
        min_combined_velocity = session['min_combined_velocity']
        velocity_range_combined = session['velocity_range_combined']
        
        personal_note = session['personal_note']
        
        # Add session data to the list for DataFrame creation
        data.append({
            'date': session_date,
            'avg_velocity_100': avg_velocity_100,
            'max_velocity_100': max_velocity_100,
            'min_velocity_100': min_velocity_100,
            'velocity_range_100': velocity_range_100,
            
            'avg_velocity_80': avg_velocity_80,
            'max_velocity_80': max_velocity_80,
            'min_velocity_80': min_velocity_80,
            'velocity_range_80': velocity_range_80,
            
            'avg_velocity_90': avg_velocity_90,
            'max_velocity_90': max_velocity_90,
            'min_velocity_90': min_velocity_90,
            'velocity_range_90': velocity_range_90,
            
            'avg_combined_velocity': avg_combined_velocity,
            'max_combined_velocity': max_combined_velocity,
            'min_combined_velocity': min_combined_velocity,
            'velocity_range_combined': velocity_range_combined,
            
            'personal_note': personal_note
        })
    
    df = pd.DataFrame(data)
    return df


def plot_session_performance_heatmap(df, heatmap_image_path):
    """
    Plots a heatmap of session performance metrics over time and saves it to the specified path.
    Args:
        df (pd.DataFrame): The DataFrame containing session data with columns for metrics (avg_velocity, max_velocity, etc.) and session dates.
        heatmap_image_path (str): The path where the heatmap image will be saved.
    """
    # Reverse the DataFrame order to have the newest session date on the left
    df = df.iloc[::-1]  # Reverse row order

    # Set session date as index for proper row labels in the heatmap
    df.set_index('date', inplace=True)
    
    # Select the metrics we want to display in the heatmap
    metrics_df = df[
        [
            'max_velocity_100', 'avg_velocity_100', 'min_velocity_100',
            'max_velocity_90', 'avg_velocity_90', 'min_velocity_90',
            'max_velocity_80', 'avg_velocity_80', 'min_velocity_80',
            'max_combined_velocity', 'avg_combined_velocity', 'min_combined_velocity'
        ]
    ]

    # Transpose to flip axes (dates on y-axis, metrics on x-axis)
    heatmap_data = metrics_df.T

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title("Session Performance Heatmap")
    plt.ylabel("Session Date")
    plt.xlabel("Metrics")
    plt.yticks(rotation=0)  # Keep metrics labels horizontal for clarity
    plt.xticks(rotation=45)  # Rotate the dates for better readability
    plt.tight_layout()

    # Save the heatmap image
    plt.savefig(heatmap_image_path)
    plt.close()  # Close the plot to free up memory


def plot_max_velocity_line_graph(df, line_graph_image_path):
    """
    Plots a line graph of the max velocity for each session and saves it to the specified path.
    
    Args:
        df (pd.DataFrame): The DataFrame containing session data with columns for 'date' and 'max_velocity'.
        line_graph_image_path (str): The path where the line graph image will be saved.
    """
    # Reverse the DataFrame order to have the newest session date on the left
    df = df.iloc[::-1]  # Reverse row order

    # Plot the line graph
    plt.figure(figsize=(12, 8))
    plt.plot(df['date'], df['max_velocity_100'], marker='o', linestyle='-', color='r', label="Max Velocity 100% Intent")
    plt.plot(df['date'], df['max_velocity_90'], marker='o', linestyle='-', color='g', label="Max Velocity 90% Intent")
    plt.plot(df['date'], df['max_velocity_80'], marker='o', linestyle='-', color='b', label="Max Velocity 80% Intent")
    plt.plot(df['date'], df['max_combined_velocity'], marker='o', linestyle='--', color='purple', label="Max Combined Velocity")

    plt.title("Max Velocity Over Time by Intent")
    plt.xlabel("Session Date")
    plt.ylabel("Max Velocity (mph)")
    plt.xticks(rotation=45)  # Rotate the dates for better readability
    plt.legend(loc="upper left")  # Add a legend to distinguish the lines
    plt.tight_layout()

    # Save the line graph image
    plt.savefig(line_graph_image_path)
    plt.close()  # Close the plot to free up memory


def prepare_scatter_data(player_sessions):
    """
    Prepares data for scatter plot of all velocities by intent and session date.

    Args:
        player_sessions (dict): Processed player sessions containing velocities and session data.

    Returns:
        pd.DataFrame: DataFrame with intent, velocity, and session date for plotting.
    """
    data = []

    # Loop through each player's session data
    for player_name, player_data in player_sessions.items():
        for session in player_data['session_metrics']:
            session_date = session['date']

            # Add 80% intent velocities
            for velocity in session['velocities_80']:
                data.append({'intent': '80%', 'velocity': velocity, 'session_date': session_date})

            # Add 90% intent velocities
            for velocity in session['velocities_90']:
                data.append({'intent': '90%', 'velocity': velocity, 'session_date': session_date})

            # Add 100% intent velocities
            for velocity in session['velocities_100']:
                data.append({'intent': '100%', 'velocity': velocity, 'session_date': session_date})

        
    # Create a DataFrame for plotting
    scatter_data = pd.DataFrame(data)
    return scatter_data


def plot_scatter_interactive(scatter_data):
    """
    Creates an interactive scatter plot of velocities by intent and session date.

    Args:
        scatter_data (pd.DataFrame): DataFrame with intent, velocity, and session date.

    Returns:
        str: HTML representation of the scatter plot.
    """
    fig = px.scatter(
        scatter_data,
        x='intent',
        y='velocity',
        color='session_date',  # Color-code by session date
        hover_data=['session_date'],  # Add session date to hover tooltip
        title="Velocities by Intent Level",
        labels={'intent': 'Intent Level', 'velocity': 'Velocity (mph)'}
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8))  # Customize marker size and transparency
    return fig.to_html(full_html=False)  # Return HTML for embedding


def generate_velocity_leaderboard(session_metrics):
    """
    Generates a leaderboard of the top 5 players with the highest maximum velocities.

    Args:
    session_metrics (dict): Dictionary containing player session data.

    Returns:
    list of tuples: List of the top 5 players with the greatest max velocity.
    """
    # Extract the player names and their max velocities
    leaderboard = []
    
    for player_name, data in session_metrics.items():
        max_velocity = data['all_time_max_velocity']
        leaderboard.append((player_name, max_velocity))

    # Sort by max velocity in descending order and get the top 5 players
    leaderboard_sorted = sorted(leaderboard, key=lambda x: x[1], reverse=True)[:5]
    
    return leaderboard_sorted


# Define a simple route
@app.route('/')
def home():
    sheet_data = get_google_sheet_data()

    # Get the unique player names for the dropdown
    player_names = get_unique_player_names(sheet_data)
    
    # Create a list of tuples with (normalized_name, hyphenated_name)
    player_name_options = [(player, player.replace(' ', '-').lower()) for player in player_names]
    
    player_sessions = process_player_data(sheet_data)

    session_metrics = calculate_session_metrics(player_sessions)

    # Generate velocity leaderboard
    leaderboard = generate_velocity_leaderboard(session_metrics)

    # Pass the player_name_options to the template
    return render_template('index.html', player_name_options=player_name_options, leaderboard=leaderboard)


@app.route('/report')
def report():
    # Get the player name from the query parameter
    player_name = request.args.get('player_name')

    if not player_name:
        return render_template('error.html', message="No player selected. Please go back and choose a player.")


    # Convert hyphens back to spaces
    player_name_normalized = player_name.replace('-', ' ').title()  # Normalize name

    
    # Generate or retrieve the cached report
    report_data = generate_player_report(player_name_normalized)


    if not report_data:  # Handle missing or invalid player data
        error_message = "The selected player was not found in the data. Please try again. Contact Matthew or Richie if you think this is a mistake."
        return render_template('error.html', error_message=error_message)
    

    # Pass the pre-computed data to the template
    return render_template(
        'report.html',
        player_name=player_name_normalized,
        session_metrics=report_data['player_data'],
        heatmap_image_path=report_data['heatmap_image_path'],
        player_profile=report_data['player_profile'],
        scatter_plot_html=report_data['scatter_plot_html']
    )

@app.route('/clear_cache')
def clear_cache():
    cache.clear()
    return "Cache cleared!"


@app.route('/update_data', methods=['POST'])
def update_data():
    """
    Endpoint to clear caches when data is updated.
    """
    cache.clear()  # Clear all cached data
    return "Data updated and cache cleared!", 200


if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)