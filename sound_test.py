import pygame
import time

# Initialize pygame mixer
pygame.mixer.init()

# Load the sound
sound = pygame.mixer.Sound("alarm.wav")

# Play the sound on loop
print("Playing alarm sound...")
sound.play(loops=-1)

# Keep the program alive while the sound is playing
time.sleep(10)  # keep alive for 10 seconds

# Stop playback
sound.stop()
print("Alarm stopped.")
