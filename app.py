from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from branca.colormap import linear
from ipyleaflet import (
    CircleMarker,
    DivIcon,
    FullScreenControl,
    LayersControl,
    Map,
    Marker,
    Popup,
    ScaleControl,
    WidgetControl,
    basemaps,
)
from ipywidgets import HTML
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget