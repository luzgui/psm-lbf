"""
2D rendering of the level-based foraging domain.
"""

import math
import os
import sys
import numpy as np

try:
    import pyglet
    from pyglet.gl import *
except ImportError:
    raise ImportError(
        """
        Cannot import pyglet.
        HINT: Install pyglet via 'pip install pyglet'.
        Alternatively, install all Gym dependencies using 'pip install gym[all]'.
        """
    )

# Define constants and colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)

_BACKGROUND_COLOR = [c / 255 for c in _WHITE]
_GRID_COLOR = _BLACK


def get_display(spec):
    """Convert a display specification (such as :0) into a Display object."""
    if spec is None:
        return pyglet.canvas.get_display()
    elif isinstance(spec, str):
        return pyglet.canvas.Display(spec)
    else:
        raise ValueError(f"Invalid display specification: {spec}. Must be a string or None.")


class Viewer:
    def __init__(self, world_size):
        self.rows, self.cols = world_size
        self.grid_size = 50
        self.icon_size = 20

        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 1 + self.rows * (self.grid_size + 1)

        self.window = pyglet.window.Window(width=self.width, height=self.height)
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        script_dir = os.path.dirname(__file__)
        resource_path = os.path.join(script_dir, "icons")
        pyglet.resource.path = [resource_path]
        pyglet.resource.reindex()

        try:
            self.img_storage = pyglet.resource.image("storage.png")
            self.img_network = pyglet.resource.image("network.png")
            self.img_agent = pyglet.resource.image("agent.png")
        except pyglet.resource.ResourceNotFoundException as e:
            print(f"Resource loading error: {e}")
            print(f"Expected resource path: {resource_path}")
            raise

    def close(self):
        """Closes the rendering window."""
        if self.isopen:
            self.window.close()
        self.isopen = False

    def window_closed_by_user(self):
        """Handles the event when the window is manually closed."""
        self.isopen = False

    def render(self, env, return_rgb_array=False):
        """Render the environment."""
        glClearColor(*_BACKGROUND_COLOR, 1.0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_resources(env)
        self._draw_players(env)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, :3]
            return arr

        self.window.flip()
        return self.isopen

    def _draw_grid(self):
        """Draw the grid on the environment."""
        batch = pyglet.graphics.Batch()
        for r in range(self.rows + 1):
            batch.add(
                2,
                pyglet.gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        0,
                        (self.grid_size + 1) * r + 1,
                        (self.grid_size + 1) * self.cols,
                        (self.grid_size + 1) * r + 1,
                    ),
                ),
                ("c3B", (*_GRID_COLOR, *_GRID_COLOR)),
            )
        for c in range(self.cols + 1):
            batch.add(
                2,
                pyglet.gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * c + 1,
                        0,
                        (self.grid_size + 1) * c + 1,
                        (self.grid_size + 1) * self.rows,
                    ),
                ),
                ("c3B", (*_GRID_COLOR, *_GRID_COLOR)),
            )
        batch.draw()

    def _draw_resources(self, env):
        for x in range(env.field_size[0]):
            for y in range(env.field_size[1]):
                # Draw Storage Resource with Level
                if env.storage_layer[x, y] > 0:
                    sprite = pyglet.sprite.Sprite(
                        self.img_storage,
                        (self.grid_size + 1) * y,
                        self.height - (self.grid_size + 1) * (x + 1),
                    )
                    sprite.scale = self.grid_size / self.img_storage.width
                    sprite.draw()
                    
                    # Draw Storage Level
                    self._draw_badge(x, y, env.storage_layer[x, y])
    
                # Draw Network Resource with Level
                if env.network_layer[x, y] > 0:
                    sprite = pyglet.sprite.Sprite(
                        self.img_network,
                        (self.grid_size + 1) * y,
                        self.height - (self.grid_size + 1) * (x + 1),
                    )
                    sprite.scale = self.grid_size / self.img_network.width
                    sprite.draw()
                    
                    # Draw Network Level
                    self._draw_badge(x, y, env.network_layer[x, y])

    def _draw_players(self, env):
        """Draw agents on the grid."""
        for agent_id, player in zip(env.agents_id, env.players):
            x, y = player.position
            sprite = pyglet.sprite.Sprite(self.img_agent,(self.grid_size + 1) * y,
                self.height - (self.grid_size + 1) * (x + 1),)
            sprite.scale = self.grid_size / self.img_agent.width
            sprite.draw()
            self._draw_badge(x, y, player.level)

    def _draw_badge(self, row, col, energy):
        """Draw a badge (e.g., energy level) next to the agent."""
        resolution = 6
        radius = self.grid_size / 5

        badge_x = col * (self.grid_size + 1) + (3 / 4) * (self.grid_size + 1)
        badge_y = self.height - (self.grid_size + 1) * (row + 1) + (1 / 4) * (self.grid_size + 1)

        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]

        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_WHITE)
        circle.draw(GL_POLYGON)
        glColor3ub(*_BLACK)
        circle.draw(GL_LINE_LOOP)

        label = pyglet.text.Label(
            f"{round(energy)}",              
            font_name="Times New Roman",
            font_size=12,
            bold=True,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
            color=(*_BLACK, 255),
        )
        label.draw()
