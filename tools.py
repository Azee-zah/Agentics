from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import Literal

@tool
def temp_converter(value, from_unit, to_unit):
    """
    Evaluate the temperature value, convert to its equivalent and return the result.
    Use this tool when you need to convert temperature in degree celsius to degree fahrenheit or from degree fahrenheit to degree celsius

    Args:
        value: A temperature value like 14C to F or 20F to C

    Returns:
        The result as a string

    Examples:
        - "14C" returns "57.2F"
        - "40F" returns "4.44C"
    """

    try:
        temp = float(value)
        if from_unit.upper() == 'C' and to_unit.upper() == 'F':
            return round((temp * 9/5) + 32, 2)
        elif from_unit.upper() == 'F' and to_unit.upper() == 'C':
            return round((temp - 32) * 5/9, 2)
    except Exception as e:
        return f"Error encountered converting temperature: {str(e)}"
    

print("Tempeee tool created")

# test tool

convert = temp_converter.invoke({"value": "25", "from_unit": "C", "to_unit": "F"})
print(f"Converted 25C to F = {convert}")