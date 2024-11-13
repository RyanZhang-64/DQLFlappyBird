import pygame
import random
import sys
from pygame.locals import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

# Initialize Pygame
pygame.init()

# Game Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 60

# Physics Constants
GRAVITY = 0.5
JUMP_SPEED = -7
PIPE_SPEED = 3

# Game Design Constants
PIPE_SPAWN_TIME = 1500  # milliseconds
PIPE_GAP = 200  # pixels
BIRD_START_X = SCREEN_WIDTH // 3
BIRD_START_Y = SCREEN_HEIGHT // 2

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird AI')
clock = pygame.time.Clock()

# ML Training Constants
MEMORY_SIZE = 100000
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.01
LEARNING_RATE = 0.001

# Faster exploration decay
EPSILON_DECAY = 0.997  # Was 0.995

# Smaller batch size initially
BATCH_SIZE = 32  # Was 64

# More frequent target network updates
TARGET_UPDATE_FREQUENCY = 50  # Was 100

class NetworkVisualizer:
    def __init__(self, screen_width, screen_height):
        self.network_surface = pygame.Surface((screen_width, screen_height))
        self.width = screen_width
        self.height = screen_height
        
        # Node positioning
        self.input_layer_y = 50
        self.hidden_layer_y = self.height // 2
        self.output_layer_y = self.height - 50
        
        # Enhanced visual parameters
        self.node_radius = 5
        self.active_color = (0, 255, 0)     # Bright green
        self.inactive_color = (0, 40, 0)    # Very dark green
        self.connection_width = 2
        
        self._init_node_positions()

    def _init_node_positions(self):
        """
        Calculate positions for all nodes in the network visualization.
        Returns a list of (x, y) coordinates for each layer's nodes.
        
        Layout structure:
        - Input layer (4 nodes): bird_y, bird_velocity, distance_to_pipe, pipe_height
        - Hidden layer (8 nodes): Internal processing nodes
        - Output layer (2 nodes): Jump or Don't Jump
        """
        # INPUT LAYER (4 nodes)
        input_spacing = 80  # Pixels between each input node
        # Center the input nodes horizontally by calculating starting x position
        input_start_x = self.width // 2 - (3 * input_spacing) // 2
        self.input_nodes = [
            (input_start_x + i * input_spacing, self.input_layer_y)
            for i in range(4)
        ]
        # Result example:
        # [(120, 50), (200, 50), (280, 50), (360, 50)]
        
        # HIDDEN LAYER (8 nodes)
        hidden_spacing = 40  # Smaller spacing since more nodes
        # Center the hidden nodes
        hidden_start_x = self.width // 2 - (7 * hidden_spacing) // 2
        self.hidden_nodes = [
            (hidden_start_x + i * hidden_spacing, self.hidden_layer_y)
            for i in range(8)
        ]
        # Result example:
        # [(100, 300), (140, 300), (180, 300), ... (380, 300)]
        
        # OUTPUT LAYER (2 nodes)
        output_spacing = 80  # Wide spacing for clarity
        # Center the two output nodes
        output_start_x = self.width // 2 - output_spacing // 2
        self.output_nodes = [
            (output_start_x + i * output_spacing, self.output_layer_y)
            for i in range(2)
        ]
        # Result example:
        # [(160, 550), (240, 550)]

    def _get_connection_color(self, weight, activation):
        """
        Calculate connection color based on weight and activation.
        Returns both color and alpha for visualization.
        """
        # Normalize the activation to 0-1 range
        activation_strength = abs(activation)
        normalized_strength = min(activation_strength, 1.0)
        
        if activation > 0:
            # Positive activation: Green with varying intensity
            return (0, int(normalized_strength * 255), 0)
        else:
            # Negative activation: Dark green
            return (0, int(normalized_strength * 40), 0)

    def draw_network(self, game_state, q_values, model):
        self.network_surface.fill((0, 0, 0))  # Black background
        
        # Get network weights
        with torch.no_grad():
            # Get first layer weights (input to hidden)
            input_weights = model.network[0].weight.detach().cpu().numpy()
            # Get last layer weights (hidden to output)
            output_weights = model.network[4].weight.detach().cpu().numpy()

        # Convert input state to array
        input_values = np.array([
            game_state['bird_y'],
            game_state['bird_velocity'],
            game_state['distance_to_pipe'],
            game_state['pipe_height']
        ])

        # Draw input to hidden connections
        for i, input_pos in enumerate(self.input_nodes):
            for j, hidden_pos in enumerate(self.hidden_nodes):
                weight = input_weights[j][i]
                activation = input_values[i] * weight
                color = self._get_connection_color(weight, activation)
                
                # Draw connection with varying thickness based on importance
                thickness = max(1, int(abs(activation) * self.connection_width))
                pygame.draw.line(
                    self.network_surface,
                    color,
                    input_pos,
                    hidden_pos,
                    thickness
                )

        # Calculate hidden layer activations (simplified)
        hidden_activations = np.dot(input_values, input_weights.T)
        hidden_activations = np.maximum(hidden_activations, 0)  # ReLU activation

        # Draw hidden to output connections
        for i, hidden_pos in enumerate(self.hidden_nodes):
            for j, output_pos in enumerate(self.output_nodes):
                weight = output_weights[j][i]
                activation = hidden_activations[i] * weight
                color = self._get_connection_color(weight, activation)
                
                thickness = max(1, int(abs(activation) * self.connection_width))
                pygame.draw.line(
                    self.network_surface,
                    color,
                    hidden_pos,
                    output_pos,
                    thickness
                )

        # Draw nodes with activation intensity
        # Input nodes
        for i, pos in enumerate(self.input_nodes):
            intensity = int(min(abs(input_values[i]), 1.0) * 255)
            color = (0, intensity, 0)
            pygame.draw.circle(self.network_surface, color, pos, self.node_radius)

        # Hidden nodes
        for i, pos in enumerate(self.hidden_nodes):
            intensity = int(min(abs(hidden_activations[i]), 1.0) * 255)
            color = (0, intensity, 0)
            pygame.draw.circle(self.network_surface, color, pos, self.node_radius)

        # Output nodes
        q_values_np = q_values.detach().cpu().numpy()
        for i, pos in enumerate(self.output_nodes):
            intensity = int(min(abs(q_values_np[i]), 1.0) * 255)
            color = (0, intensity, 0)
            pygame.draw.circle(self.network_surface, color, pos, self.node_radius)

        # Draw labels
        self._draw_enhanced_labels(game_state, q_values_np, input_values)
        
        return self.network_surface

    def _draw_enhanced_labels(self, game_state, q_values, input_values):
        font = pygame.font.Font(None, 20)
        
        # Input labels with names and values on separate lines
        input_names = ["Height", "Velocity", "Distance", "Gap Height"]
        
        for i, (pos, name) in enumerate(zip(self.input_nodes, input_names)):
            # Draw name on first line
            name_text = font.render(name, True, (255, 255, 255))
            name_rect = name_text.get_rect(centerx=pos[0], bottom=pos[1] - 15)
            self.network_surface.blit(name_text, name_rect)
            
            # Draw value on second line
            value_text = font.render(f"{input_values[i]:.2f}", True, (255, 255, 255))
            value_rect = value_text.get_rect(centerx=pos[0], top=pos[1] + 15)
            self.network_surface.blit(value_text, value_rect)
        
        # Output labels with names and probabilities on separate lines
        actions = ["Don't Jump", "Jump"]
        # Convert Q-values to probabilities using softmax
        exp_qvals = np.exp(q_values)
        probabilities = exp_qvals / np.sum(exp_qvals)
        
        for i, (pos, action) in enumerate(zip(self.output_nodes, actions)):
            # Draw action name above
            action_text = font.render(action, True, (255, 255, 255))
            action_rect = action_text.get_rect(centerx=pos[0], top=pos[1] + 15)
            self.network_surface.blit(action_text, action_rect)
            
            # Draw probability below
            prob_text = font.render(f"{probabilities[i]:.2%}", True, (255, 255, 255))
            prob_rect = prob_text.get_rect(centerx=pos[0], top=pos[1] + 35)
            self.network_surface.blit(prob_text, prob_rect)

        # Optionally add hidden layer labels (just numbers)
        for i, pos in enumerate(self.hidden_nodes):
            hidden_text = font.render(str(i+1), True, (255, 255, 255))
            hidden_rect = hidden_text.get_rect(centerx=pos[0], top=pos[1] + 15)
            self.network_surface.blit(hidden_text, hidden_rect)

class TrainingManager:
    def __init__(self, save_dir='training_data'):
        self.save_dir = save_dir
        self.episode_scores = []
        self.episode_losses = []
        self.best_score = float('-inf')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    def log_episode(self, episode, score, avg_loss, epsilon):
        self.episode_scores.append(score)
        self.episode_losses.append(avg_loss)
        
        if score > self.best_score:
            self.best_score = score
            return True
        return False
    
    def save_training_data(self):
        training_data = {
            'scores': self.episode_scores,
            'losses': self.episode_losses,
            'best_score': self.best_score
        }
        
        filename = f"{self.save_dir}/training_data_{self.timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(training_data, f)
    
    def plot_training_progress(self):
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(self.episode_scores)
        plt.title('Episode Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        plt.subplot(3, 1, 2)
        plt.plot(self.episode_losses)
        plt.title('Average Loss per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        # Plot moving average of scores
        window_size = 100
        moving_avg = np.convolve(self.episode_scores, 
                                np.ones(window_size)/window_size, 
                                mode='valid')
        plt.subplot(3, 1, 3)
        plt.plot(moving_avg)
        plt.title(f'Moving Average of Scores (Window Size: {window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_progress_{self.timestamp}.png")
        plt.close()

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class FlappyAgent:
    def __init__(self, state_size=4, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-Network and target network
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training parameters
        self.learning_rate = LEARNING_RATE
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.batch_size = BATCH_SIZE
        
        # Exploration parameters
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.gamma = GAMMA
        
    def get_state(self, game_state):
        # Convert dictionary to tensor
        return torch.FloatTensor([
            game_state['bird_y'],
            game_state['bird_velocity'],
            game_state['distance_to_pipe'],
            game_state['pipe_height']
        ])
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            # No need to unsqueeze since state is already a tensor
            action_values = self.q_network(state.unsqueeze(0))
            return torch.argmax(action_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        # Store everything as tensors
        self.memory.append((
            state,  # Already a tensor
            action,
            torch.FloatTensor([reward]),
            next_state,  # Already a tensor
            torch.FloatTensor([done])
        ))
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Stack tensors instead of converting lists
        states = torch.stack([experience[0] for experience in batch])
        actions = torch.LongTensor([[experience[1]] for experience in batch])
        rewards = torch.stack([experience[2] for experience in batch])
        next_states = torch.stack([experience[3] for experience in batch])
        dones = torch.stack([experience[4] for experience in batch])
        
        # Get current Q values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Get next Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
        # Calculate target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calculate loss and update network
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def save(self, filename):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class Bird:
    def __init__(self):
        self.image = pygame.image.load('Downloads/flappybird/bird.png').convert_alpha()
        self.rect = self.image.get_rect()
        self.reset()
        
    def reset(self):
        self.rect.x = SCREEN_WIDTH // 3
        self.rect.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        
    def jump(self):
        self.velocity = JUMP_SPEED
        
    def update(self):
        self.velocity += GRAVITY
        self.rect.y += self.velocity
        
        if self.rect.top < 0:
            self.rect.top = 0
            self.velocity = 0
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT
            self.velocity = 0
            
    def draw(self, surface):
        surface.blit(self.image, self.rect)

class Pipe:
    def __init__(self, x):
        self.image = pygame.image.load('Downloads/flappybird/pipe.png').convert_alpha()
        self.image_inverted = pygame.transform.flip(self.image, False, True)
        
        self.rect_top = self.image_inverted.get_rect()
        self.rect_bottom = self.image.get_rect()
        
        pipe_height = random.randint(100, SCREEN_HEIGHT - PIPE_GAP - 100)
        self.rect_top.bottomleft = (x, pipe_height)
        self.rect_bottom.topleft = (x, pipe_height + PIPE_GAP)
        
    def update(self):
        self.rect_top.x -= PIPE_SPEED
        self.rect_bottom.x -= PIPE_SPEED
        
    def draw(self, surface):
        surface.blit(self.image_inverted, self.rect_top)
        surface.blit(self.image, self.rect_bottom)
        
    def is_off_screen(self):
        return self.rect_top.right < 0

class Game:
    def __init__(self):
        self.background = pygame.image.load('Downloads/flappybird/background.png').convert()
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.font = pygame.font.Font('Downloads/flappybird/flappy-font.ttf', 36)
        self.game_state = 'playing'
        self.last_pipe_time = pygame.time.get_ticks()
        self.frames_alive = 0
        self.last_pipe_distance = float('inf')
        # Create a surface for the game
        self.game_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        # Initialize with first pipe
        self.pipes.append(Pipe(SCREEN_WIDTH))

    def get_game_state(self):
        state = {
            'bird_y': self.bird.rect.centery / SCREEN_HEIGHT,
            'bird_velocity': self.bird.velocity / 10.0,
            'distance_to_pipe': float('inf'),
            'pipe_height': 0.5  # Default when no pipe
        }

        # Find nearest pipe
        nearest_pipe = None
        nearest_distance = float('inf')
        
        for pipe in self.pipes:
            if pipe.rect_top.right > self.bird.rect.left:
                distance = pipe.rect_top.left - self.bird.rect.right
                if distance < nearest_distance:
                    nearest_pipe = pipe
                    nearest_distance = distance
                    state['distance_to_pipe'] = distance / SCREEN_WIDTH
                    state['pipe_height'] = pipe.rect_top.bottom / SCREEN_HEIGHT

        return state

    def step(self, action):
        # Start with small survival reward
        reward = 0.05  # Reduced base survival reward
        
        # Execute action and track vertical position
        if action == 1:
            self.bird.jump()
        
        # Track vertical movement
        previous_y = self.bird.rect.centery
        self.bird.update()
        
        # Penalize extreme vertical movements
        vertical_change = abs(self.bird.rect.centery - previous_y)
        if vertical_change > 10:  # If movement is too dramatic
            reward -= 0.1 * (vertical_change / 10)
        
        # Update pipes
        self.spawn_pipe()
        self.pipes = [pipe for pipe in self.pipes if not pipe.is_off_screen()]
        for pipe in self.pipes:
            pipe.update()
        
        # Find nearest pipe
        nearest_pipe = None
        nearest_distance = float('inf')
        for pipe in self.pipes:
            if pipe.rect_top.right > self.bird.rect.left:
                distance = pipe.rect_top.left - self.bird.rect.right
                if distance < nearest_distance:
                    nearest_pipe = pipe
                    nearest_distance = distance
        
        # Reward for good positioning relative to pipe
        if nearest_pipe:
            pipe_center_y = (nearest_pipe.rect_top.bottom + nearest_pipe.rect_bottom.top) / 2
            vertical_distance = abs(self.bird.rect.centery - pipe_center_y)
            
            # Stronger penalty for being far from pipe center
            reward -= (vertical_distance / SCREEN_HEIGHT) * 0.2
            
            # Extra penalty for being above top pipe or below bottom pipe
            if self.bird.rect.centery < nearest_pipe.rect_top.bottom:
                reward -= 0.3
            elif self.bird.rect.centery > nearest_pipe.rect_bottom.top:
                reward -= 0.3
                
            # Only reward horizontal approach if vertically aligned
            if vertical_distance < PIPE_GAP/2:  # If reasonably aligned with gap
                if nearest_distance < self.last_pipe_distance:
                    reward += 0.05  # Reduced approach reward
        
        # Reward for passing pipes
        for pipe in self.pipes:
            if pipe.rect_top.right < self.bird.rect.left and not hasattr(pipe, 'scored'):
                pipe.scored = True
                self.score += 1
                reward += 2.0  # Increased success reward
        
        # Check if game over
        done = False
        if self.check_collisions() or self.bird.rect.bottom >= SCREEN_HEIGHT:
            reward = -2.0  # Increased death penalty
            done = True
        
        # Update last pipe distance for next iteration
        self.last_pipe_distance = nearest_distance
        return self.get_game_state(), reward, done
        
    def reset(self):
        self.bird.reset()
        self.pipes.clear()  # Clear existing pipes
        self.score = 0
        self.frames_alive = 0
        self.game_state = 'playing'
        self.last_pipe_time = pygame.time.get_ticks()
        # Always create initial pipe
        self.pipes.append(Pipe(SCREEN_WIDTH))
        return self.get_game_state()
        
    def spawn_pipe(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_pipe_time > PIPE_SPAWN_TIME:
            self.pipes.append(Pipe(SCREEN_WIDTH))
            self.last_pipe_time = current_time
            
    def check_collisions(self):
        for pipe in self.pipes:
            if (self.bird.rect.colliderect(pipe.rect_top) or 
                self.bird.rect.colliderect(pipe.rect_bottom)):
                return True
        return False
    
    def update_score(self):
        for pipe in self.pipes:
            if pipe.rect_top.right < self.bird.rect.left and not hasattr(pipe, 'scored'):
                self.score += 1
                pipe.scored = True
                
    def draw(self):
        # Draw to game_surface instead of screen
        self.game_surface.blit(self.background, (0, 0))
        
        for pipe in self.pipes:
            pipe.draw(self.game_surface)
            
        self.bird.draw(self.game_surface)
        
        score_text = self.font.render(f'Score: {self.score}', True, BLACK)
        self.game_surface.blit(score_text, (10, 10))
        
        return self.game_surface  # Return the surface instead of calling display.flip()

def main():
    # Initialize with double width for network visualization
    screen = pygame.display.set_mode((SCREEN_WIDTH * 2, SCREEN_HEIGHT))
    pygame.display.set_caption('Flappy Bird AI')
    
    game = Game()
    agent = FlappyAgent()
    network_vis = NetworkVisualizer(SCREEN_WIDTH, SCREEN_HEIGHT)
    
    clock = pygame.time.Clock()
    episode = 0
    
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        
        # Get current state and action
        state = game.get_game_state()
        state_tensor = agent.get_state(state)
        
        # Get Q-values for visualization
        with torch.no_grad():
            q_values = agent.q_network(state_tensor.unsqueeze(0))[0]
        
        action = agent.act(state_tensor)
        
        # Execute action
        next_state, reward, done = game.step(action)
        
        # Remember and train
        next_state_tensor = agent.get_state(next_state)
        agent.remember(state_tensor, action, reward, next_state_tensor, done)
        loss = agent.train()
        
        # Draw everything
        screen.fill((0, 0, 0))
        
        # Draw game
        game_surface = game.draw()
        screen.blit(game_surface, (0, 0))
        
        # Draw network visualization
        network_surface = network_vis.draw_network(state, q_values, agent.q_network)
        screen.blit(network_surface, (SCREEN_WIDTH, 0))
        
        pygame.display.flip()
        
        if done:
            game.reset()
            episode += 1
        
        clock.tick(60)

if __name__ == '__main__':
    main()