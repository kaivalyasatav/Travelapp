import gradio as gr
import requests
import json

class TripPlanner:
    def __init__(self, ollama_base_url="http://localhost:11434"):
        self.ollama_base_url = ollama_base_url
        
    def generate_prompt(self, destination, num_days):
        return f"""Create a detailed {num_days}-day trip itinerary for {destination}. \
        Please provide a day-by-day breakdown including:\
        - Morning activities and attractions\
        - Afternoon activities\
        - Evening activities and dinner suggestions\
        - Must-visit locations\
        - Local food recommendations\
        - Transportation tips\
        \n        Please write it in a natural, conversational style that's easy to read.\
        Make it detailed but friendly, as if you're suggesting this itinerary to a friend."""

    def get_itinerary(self, destination, num_days):
        prompt = self.generate_prompt(destination, num_days)
        
        try:
            # Test connection to Ollama first
            try:
                requests.get(f"{self.ollama_base_url}/api/tags")
            except requests.exceptions.ConnectionError:
                return "Error: Cannot connect to Ollama. Please ensure Ollama is running (ollama run mistral)"

            # Use the correct API endpoint for chat completion
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": "llama3.2:1b",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                response_text = response.json().get('message', {}).get('content', '')
                return self.format_output(response_text, num_days, destination)
            else:
                return (f"Error: Unable to generate itinerary. Status code: {response.status_code}\n"
                       f"Make sure you have pulled the mistral model using 'ollama pull mistral'")
                
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}\nPlease ensure Ollama is running on {self.ollama_base_url}"

    def format_output(self, text, num_days, destination):
        formatted_output = [
            f"Your {num_days}-Day Trip to {destination.title()}\n",
            "=" * 50
        ]

        days = []
        current_day = []

        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            day_starts = [f"Day {i}:" for i in range(1, num_days + 1)] + [f"Day {i}" for i in range(1, num_days + 1)]

            if any(line.startswith(day) for day in day_starts):
                if current_day:
                    days.append('\n'.join(current_day))
                current_day = [line]
            else:
                current_day.append(line)

        if current_day:
            days.append('\n'.join(current_day))

        if len(days) != num_days:
            text_chunks = text.split('\n\n')
            days = []
            current_chunk = []
            chunk_size = len(text_chunks) // num_days

            for i, chunk in enumerate(text_chunks):
                current_chunk.append(chunk)
                if (i + 1) % chunk_size == 0 and len(days) < num_days:
                    days.append('\n'.join(current_chunk))
                    current_chunk = []

        for i, day_content in enumerate(days, 1):
            formatted_output.extend([
                f"\nDay {i}\n",
                "-" * 50,
                day_content.strip(),
                "\n"
            ])

        formatted_output.extend([
            "\nTravel Tips:\n",
            "• Remember to check opening hours for attractions",
            "• Keep local emergency numbers handy",
            "• Respect local customs and traditions",
            "• Stay hydrated and carry weather-appropriate gear",
            "• Have a fantastic trip!"
        ])

        return '\n'.join(formatted_output)


def interact_with_trip_planner(destination, num_days):
    try:
        num_days = int(num_days)
        if num_days <= 0:
            return "Please enter a positive number of days."
    except ValueError:
        return "Invalid number of days. Please enter a valid number."

    planner = TripPlanner()
    return planner.get_itinerary(destination, num_days)


def main():
    iface = gr.Interface(
        fn=interact_with_trip_planner,
        inputs=[
            gr.Textbox(label="Destination", placeholder="Enter your travel destination"),
            gr.Textbox(label="Number of Days", placeholder="Enter number of days")
        ],
        outputs=gr.Textbox(label="Travel Itinerary"),
        title="Trip Planner",
        description="Generate a detailed trip itinerary for your destination."
    )

    iface.launch(share=True)


if __name__ == "__main__":
    main()