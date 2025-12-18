import cv2
import numpy as np
import time

class BouncingBall:
    """Simple state holder for bouncing ball animation."""
    def __init__(self, size=(640, 480), radius=30, color=(0,0,255), velocity=(6,6)):
        self.width, self.height = size
        self.radius = radius
        self.color = color
        self.vx, self.vy = velocity
        self.x = self.width // 2
        self.y = self.height // 2

    def next_frame(self):
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Bounce
        if self.x - self.radius <= 0 or self.x + self.radius >= self.width:
            self.vx *= -1
            self.x = max(self.radius, min(self.x, self.width - self.radius))
        if self.y - self.radius <= 0 or self.y + self.radius >= self.height:
            self.vy *= -1
            self.y = max(self.radius, min(self.y, self.height - self.radius))
            
        # Draw
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.circle(frame, (self.x, self.y), self.radius, self.color, -1)
        return frame

class BoxWithText:
    """Simple state holder for box animation."""
    def __init__(self, text="DustyCam", size=(640, 480), box_size=(100,50), velocity=(5,5)):
        self.text = text
        self.width, self.height = size
        self.box_w, self.box_h = box_size
        self.vx, self.vy = velocity
        self.x = (self.width - self.box_w) // 2
        self.y = (self.height - self.box_h) // 2

    def next_frame(self):
        self.x += self.vx
        self.y += self.vy
        
        if self.x <= 0 or self.x + self.box_w >= self.width:
            self.vx *= -1
            self.x = max(0, min(self.x, self.width - self.box_w))
        if self.y <= 0 or self.y + self.box_h >= self.height:
            self.vy *= -1
            self.y = max(0, min(self.y, self.height - self.box_h))
            
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.box_w, self.y + self.box_h), (255, 255, 255), -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        (text_w, text_h), _ = cv2.getTextSize(self.text, font, text_scale, 1)
        text_x = self.x + (self.box_w - text_w) // 2
        text_y = self.y + (self.box_h + text_h) // 2
        cv2.putText(frame, self.text, (text_x, text_y), font, text_scale, (0, 0, 0), 1, cv2.LINE_AA)
        return frame
