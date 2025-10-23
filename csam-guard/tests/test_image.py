import base64
from csam_guard.guard import CSAMGuard, DEFAULT_CONFIG

def test_image_paths():
    g = CSAMGuard(DEFAULT_CONFIG)
    mock_safe_image = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAABNSURBVHhe7c4xCoAwEATB3v+j3d3d3VYT8j6QEG3bDzzvBzSaTdPcNLtgbtPcpblNc5vmNs1tmts0t2lu09ymuU1zm+Y2zW2a2zS3aW7T3AAAAAD//wMA3kHWkQAAAABJRU5ErkJggg=="
    )
    d = g.assess_image(image_data=mock_safe_image)
    assert d.allow
