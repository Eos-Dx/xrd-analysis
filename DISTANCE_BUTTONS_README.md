# Configurable Distance Buttons Feature

## Overview

The EOSDXDC application now supports configurable distance buttons that replace the previous "Add distance" button with text field. Instead of typing distance values, you can now have multiple predefined buttons that append specific text to your filename when clicked.

## Configuration

### Location
The distance buttons are configured in the global configuration file:
```
src/hardware/eosdxdc/resources/config/global.json
```

### Format
Add a `distance_buttons` array to your config file:

```json
{
  "distance_buttons": [
    {
      "text": "+2cm",
      "append_text": "_2cm"
    },
    {
      "text": "+17cm",
      "append_text": "_17cm"
    }
  ]
}
```

### Properties
- `text`: The button label displayed in the UI
- `append_text`: The text that gets appended to the filename when clicked

## Examples

### Basic Distance Buttons (Current Default)
```json
"distance_buttons": [
  {"text": "+2cm", "append_text": "_2cm"},
  {"text": "+17cm", "append_text": "_17cm"}
]
```

### Multiple Distance Options
```json
"distance_buttons": [
  {"text": "+5mm", "append_text": "_5mm"},
  {"text": "+1cm", "append_text": "_1cm"},
  {"text": "+2cm", "append_text": "_2cm"},
  {"text": "+5cm", "append_text": "_5cm"},
  {"text": "+10cm", "append_text": "_10cm"}
]
```

### Position-Based Labels
```json
"distance_buttons": [
  {"text": "Near", "append_text": "_near"},
  {"text": "Mid", "append_text": "_mid"},
  {"text": "Far", "append_text": "_far"}
]
```

### Setup-Specific Buttons
```json
"distance_buttons": [
  {"text": "SAXS", "append_text": "_SAXS_1720mm"},
  {"text": "WAXS", "append_text": "_WAXS_20mm"}
]
```

## Behavior

1. **Dynamic Creation**: Buttons are created dynamically based on your configuration
2. **Filename Appending**: Clicking a button appends its `append_text` to the current filename
3. **No Limit**: You can configure as many buttons as needed
4. **Fallback**: If no configuration is found, defaults to "+2cm" and "+17cm" buttons

## Usage

1. Configure your desired buttons in `global.json`
2. Restart the EOSDXDC application
3. In the Measurements tab, you'll see your configured buttons
4. Enter your base filename in the "File Name" field
5. Click any distance button to append its text to the filename
6. The filename field will be updated automatically

## Migration from Old System

The previous system used:
- One "Add distance" button
- One text field for entering distance values

The new system provides:
- Multiple predefined distance buttons
- No manual text entry required
- Consistent naming conventions
- Faster workflow

## Troubleshooting

If distance buttons don't appear:
1. Check that `global.json` syntax is valid JSON
2. Ensure `distance_buttons` is an array of objects
3. Verify each button has both `text` and `append_text` properties
4. Restart the application after config changes

If buttons appear but don't work:
1. Check that the filename field exists and is accessible
2. Verify the measurement tab is properly loaded
3. Look for error messages in the application log
