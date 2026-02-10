import hx_data_schema as hx
from data_schema.data_schema_utilities import thousands_format, percent_format

def adverse_weather():
    return(
        hx.Structure(
            children={
                "adverse_weather_daily": adverse_weather_daily(),
                "adverse_weather_expanding": adverse_weather_expanding()
            }
        )
    )

def adverse_weather_calculations():
    """
    Defines the data schema for adverse weather calculation outputs.

    This schema includes fields for weather probabilities, thresholds, risk drivers,
    and the final calculated rate and loss for adverse weather perils.

    Returns:
        dict: A dictionary of schema components for adverse weather calculations.
    """
    return(
        {
                "adverse_weather_severity":hx.Float(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Severity",
                        "format":{"mantissa": 6}
                    }
                ),
                "event_duration": hx.Int(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Event Duration"
                    }
                ),
                "rain_threshold":hx.Float(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Rain Threshold",
                        "format":{"mantissa": 1}
                    }
                ),
                "max_wind_threshold":hx.Float(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Max Wind Threshold",
                        "format":{"mantissa": 1}
                    }
                ),
                "max_gust_threshold":hx.Float(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Max Gust Threshold",
                        "format":{"mantissa": 1}
                    }
                ), 
                "lightning_threshold":hx.Str(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Lightning Threshold"
                    }
                ),
                "requires_cumulative":hx.Bool(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Is Cumulative Concern?"
                    }
                ),
                "cumulative_rain_days":hx.Int(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Cumulative Rain Days",
                        "format":thousands_format(0)
                    }
                ),
                "cumulative_rain_threshold":hx.Float(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Cumulative Rain Threshold",
                        "format":{"mantissa": 1}
                    }
                ),
                "rain_prob":hx.Float(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Rain Probability",
                        "format":percent_format(5)
                    }
                ),
                "wind_prob":hx.Float(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Wind Probability",
                        "format":percent_format(5)
                    }
                ),
                "gust_prob":hx.Float(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Gust Probability",
                        "format":percent_format(5)
                    }
                ),
                "lightning_prob":hx.Float(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Lightning Probability",
                        "format":percent_format(5)
                    }
                ),
                "cumulative_rain_prob":hx.Float(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Cumulative Rain Probability",
                        "format":percent_format(5)
                    }
                ),
                "driver":hx.Str(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Driver"
                    }
                ),
                "authority_modification":hx.Float(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Authority Modification",
                        "format":percent_format(2)
                    }
                ),
                "adverse_weather_rate":hx.Float(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Adverse Weather FGU Rate",
                        "format":percent_format(5)
                    }
                ),
                "adverse_weather_fgu_loss":hx.Float(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Adverse Weather FGU Loss",
                        "format":thousands_format(0)
                    }
                ),
                "adverse_weather_layer_rate":hx.Float(
                    mode="output",
                    async_input=["adverse_weather", "cat", "run_rating", "mourning_named_person_cross_join", "non_app_named_person_cross_join", "mourning_ncr_cross_join","cat_model_export_to_excel"],
                    view={
                        "label":"Adverse Weather Layer Rate",
                        "format":percent_format(5)
                    }
                ),
            }
        )

def adverse_weather_daily():
    """
    Defines the data schema for a list of the BEV daily api adverse weather outputs.

    Each item in the list represents a daily weather metric, including the peril,
    threshold, and measured value.

    Returns:
        hx.List: The schema for a list of daily adverse weather data.
    """

    return(
        hx.List(
            mode="output",
            async_output=["adverse_weather", "run_rating"],
            children={
                "peril": hx.Str(
                    mode="output",
                    async_output=["adverse_weather", "run_rating"],
                    view={
                        "label": "Peril"
                    }
                ),
                "index": hx.Int(
                    mode="output",
                    async_output=["adverse_weather", "run_rating"],
                    view={
                        "label": "Event Index"
                    }
                ),
                "threshold": hx.Str(
                    mode="output",
                    async_output=["adverse_weather", "run_rating"],
                    view={
                        "label": "Threshold"
                    }
                ),
                "value": hx.Float(
                    mode="output",
                    async_output=["adverse_weather", "run_rating"],
                    view={
                        "label": "Value",
                        "format":{"mantissa": 8}
                    }
                )
            }
        )
    )

def adverse_weather_expanding():
    """
    Defines the data schema for BEV API expanding-window adverse weather outputs.

    Each item represents a calculation over an expanding time window, useful
    for analyzing cumulative weather effects.

    Returns:
        hx.List: The schema for a list of expanding-window weather data.
    """
    return(
        hx.List(
            mode="output",
            async_output=["adverse_weather", "run_rating"],
            children={
                "index": hx.Int(
                    mode="output",
                    async_output=["adverse_weather", "run_rating"],
                    view={
                        "label": "Event Index"
                    }
                ),
                "window_index": hx.Int(
                    mode="output",
                    async_output=["adverse_weather", "run_rating"],
                    view={
                        "label": "Window Index"
                    }
                ),
                "peril": hx.Str(
                    mode="output",
                    async_output=["adverse_weather", "run_rating"],
                    view={
                        "label": "Peril"
                    }
                ),
                "threshold": hx.Float(
                    mode="output",
                    async_output=["adverse_weather", "run_rating"],
                    view={
                        "label": "Threshold",
                        "format":{"mantissa": 0}
                    }
                ),
                "value": hx.Float(
                    mode="output",
                    async_output=["adverse_weather", "run_rating"],
                    view={
                        "label": "Value",
                        "format":{"mantissa": 8}
                    }
                )
            }
        )
    )