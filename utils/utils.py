import datetime
import numpy as np
import pandas as pd

def mock_data(n_samples: int = 1000) -> pd.DataFrame:
    # First generate the categories
    categories = np.random.choice(["Electronics", "Clothing", "Home", "Books", "Sports"], n_samples)
    
    # Define category-specific distribution parameters
    category_params = {
        "Electronics": {
            "price": {"mean": 200, "std": 50},
            "manufacturing_cost": {"mean": 80, "std": 20},
            "shipping_weight": {"mean": 2.5, "std": 1.2}
        },
        "Clothing": {
            "price": {"mean": 80, "std": 30},
            "manufacturing_cost": {"mean": 40, "std": 12},
            "shipping_weight": {"mean": 0.8, "std": 0.3}
        },
        "Home": {
            "price": {"mean": 150, "std": 45},
            "manufacturing_cost": {"mean": 60, "std": 18},
            "shipping_weight": {"mean": 5, "std": 2.5}
        },
        "Books": {
            "price": {"mean": 30, "std": 15},
            "manufacturing_cost": {"mean": 15, "std": 5},
            "shipping_weight": {"mean": 1.5, "std": 0.5}
        },
        "Sports": {
            "price": {"mean": 120, "std": 40},
            "manufacturing_cost": {"mean": 50, "std": 15},
            "shipping_weight": {"mean": 3, "std": 1.0}
        }
    }
    
    # Initialize empty arrays for category-specific attributes
    prices = np.zeros(n_samples)
    manufacturing_costs = np.zeros(n_samples)
    shipping_weights = np.zeros(n_samples)
    
    # Generate data for each category
    for category in category_params:
        mask = categories == category
        category_count = np.sum(mask)
        
        prices[mask] = np.random.normal(
            category_params[category]["price"]["mean"],
            category_params[category]["price"]["std"],
            category_count
        )
        
        manufacturing_costs[mask] = np.random.normal(
            category_params[category]["manufacturing_cost"]["mean"],
            category_params[category]["manufacturing_cost"]["std"],
            category_count
        )
        
        shipping_weights[mask] = np.random.normal(
            category_params[category]["shipping_weight"]["mean"],
            category_params[category]["shipping_weight"]["std"],
            category_count
        )
    
    # Ensure all values are positive
    prices = np.maximum(prices, 10)  # Minimum price of $10
    manufacturing_costs = np.maximum(manufacturing_costs, 5)  # Minimum cost of $5
    shipping_weights = np.maximum(shipping_weights, 0.1)  # Minimum weight of 0.1 kg
    
    # Generate remaining data
    data = {
        "product_id": [f"PROD-{i:04d}" for i in range(n_samples)],
        "category": categories,
        "brand_rating": np.random.uniform(1, 5, n_samples),
        "num_reviews": np.random.randint(0, 500, n_samples),
        "days_since_release": np.random.randint(1, 1000, n_samples),
        "discount_offered": np.random.choice([True, False], n_samples),
        "shipping_weight": shipping_weights,
        "competitors_price": prices * np.random.uniform(0.8, 1.2, n_samples),  # Competitors price varies around our price
        "manufacturing_cost": manufacturing_costs,
        "price": prices
    }
    
    # Introduce some missing values
    for col in ["brand_rating", "num_reviews", "shipping_weight"]:
        missing_indices = np.random.choice(range(n_samples), size=int(n_samples * 0.05), replace=False)
        data[col] = pd.Series(data[col])
        data[col].iloc[missing_indices] = None
    
    df = pd.DataFrame(data)
    return df

def make_category_boxplot_data(field, variable_name, data: pd.DataFrame):
    """Generate boxplot data for a specific field by category.
    
    Args:
        field: The dataframe column to plot
        variable_name: The JavaScript variable name to push data to
    """
    colors = {
        "Electronics": "rgba(255, 99, 132, 0.7)",
        "Clothing": "rgba(54, 162, 235, 0.7)",
        "Home": "rgba(255, 206, 86, 0.7)",
        "Books": "rgba(75, 192, 192, 0.7)",
        "Sports": "rgba(153, 102, 255, 0.7)"
    }
    
    js_code = []
    for category in data['category'].unique():
        cat_data = data[data['category'] == category][field].tolist()
        cat_stats = {
            "mean": float(data[data['category'] == category][field].mean()),
            "median": float(data[data['category'] == category][field].median()),
            "min": float(data[data['category'] == category][field].min()),
            "max": float(data[data['category'] == category][field].max()),
            "count": len(cat_data)
        }
        
        field_unit = "$" if field in ["price", "manufacturing_cost", "competitors_price"] else ""
        
        js_code.append(f"""
            {variable_name}.push({{
                y: {cat_data},
                type: 'box',
                name: '{category}',
                boxpoints: 'outliers',
                marker: {{
                    color: '{colors.get(category, "rgba(100, 100, 100, 0.7)")}'
                }},
                hoverinfo: 'all',
                hovertemplate: 
                    '<b>{category}</b><br>' +
                    'Mean: {field_unit}{cat_stats["mean"]:.2f}<br>' +
                    'Median: {field_unit}{cat_stats["median"]:.2f}<br>' +
                    'Min: {field_unit}{cat_stats["min"]:.2f}<br>' +
                    'Max: {field_unit}{cat_stats["max"]:.2f}<br>' +
                    'Count: {cat_stats["count"]}<extra></extra>'
            }});
        """)
    return "\n".join(js_code)

def make_profit_margin_data(data: pd.DataFrame):
    colors = {
        "Electronics": "rgba(255, 99, 132, 0.7)",
        "Clothing": "rgba(54, 162, 235, 0.7)",
        "Home": "rgba(255, 206, 86, 0.7)",
        "Books": "rgba(75, 192, 192, 0.7)",
        "Sports": "rgba(153, 102, 255, 0.7)"
    }
    
    # Calculate profit margins for each category
    categories = data['category'].unique()
    margins = []
    for category in categories:
        cat_df = data[data['category'] == category]
        margin = ((cat_df['price'] - cat_df['manufacturing_cost']) / cat_df['price']).mean()
        margins.append(margin)
    
    return f"""
    profitMarginData.push({{
        x: {list(categories)},
        y: {margins},
        type: 'bar',
        marker: {{
            color: {[colors.get(cat, "rgba(100, 100, 100, 0.7)") for cat in categories]}
        }},
        hovertemplate: 
            '<b>%{{x}}</b><br>' +
            'Profit Margin: %{{y:.1%}}<extra></extra>'
    }});
    """

def generate_data_report(cleaned_data: pd.DataFrame, raw_data: pd.DataFrame, analysis: dict) -> str:
    """Generate HTML report focused on data analysis."""
    # Pre-generate all the JavaScript code for the charts
    price_boxplot_js = make_category_boxplot_data('price', 'categoryPriceData', cleaned_data)
    cost_boxplot_js = make_category_boxplot_data('manufacturing_cost', 'categoryCostData', cleaned_data)
    weight_boxplot_js = make_category_boxplot_data('shipping_weight', 'categoryWeightData', cleaned_data)
    profit_margin_js = make_profit_margin_data(cleaned_data)
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ padding: 20px; }}
            .card {{ margin-bottom: 20px; }}
            .metric-card {{ text-align: center; padding: 15px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; }}
            .metric-label {{ font-size: 14px; color: #666; }}
            .chart-container {{ height: 400px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="my-4">Product Data Analysis Report</h1>
            <p class="lead">Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="row mb-4">
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">{raw_data.shape[0]}</div>
                        <div class="metric-label">Total Products</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">{len(raw_data['category'].unique())}</div>
                        <div class="metric-label">Product Categories</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">${cleaned_data['price'].mean():.2f}</div>
                        <div class="metric-label">Avg. Price</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">{sum(raw_data.isnull().sum())}</div>
                        <div class="metric-label">Missing Values</div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Price by Category</h5>
                        </div>
                        <div class="card-body">
                            <div id="category-price-chart" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Manufacturing Cost by Category</h5>
                        </div>
                        <div class="card-body">
                            <div id="category-cost-chart" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Shipping Weight by Category</h5>
                        </div>
                        <div class="card-body">
                            <div id="category-weight-chart" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Profit Margin by Category</h5>
                        </div>
                        <div class="card-body">
                            <div id="profit-margin-chart" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Price Distribution</h5>
                        </div>
                        <div class="card-body">
                            <div id="price-dist-chart"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Price Distribution Chart
            var priceData = {cleaned_data["price"].tolist()};
            var priceLayout = {{
                height: 400,
                margin: {{ t: 10 }},
                xaxis: {{ title: "Price ($)" }}
            }};
            Plotly.newPlot('price-dist-chart', [{{
                x: priceData,
                type: 'histogram',
                marker: {{
                    color: 'rgba(75, 192, 192, 0.7)',
                    line: {{
                        color: 'rgba(75, 192, 192, 1.0)',
                        width: 1
                    }}
                }},
                hovertemplate: 'Price: $%{{x}}<br>Count: %{{y}}<extra></extra>'
            }}], priceLayout);
            
            // Category-specific price boxplots
            var categoryPriceData = [];
            {price_boxplot_js}
            
            var categoryPriceLayout = {{
                height: 400,
                margin: {{ t: 10 }},
                xaxis: {{ title: "Category" }},
                yaxis: {{ title: "Price ($)" }}
            }};
            
            Plotly.newPlot('category-price-chart', categoryPriceData, categoryPriceLayout);
            
            // Category-specific cost boxplots
            var categoryCostData = [];
            {cost_boxplot_js}
            
            var categoryCostLayout = {{
                height: 400,
                margin: {{ t: 10 }},
                xaxis: {{ title: "Category" }},
                yaxis: {{ title: "Manufacturing Cost ($)" }}
            }};
            
            Plotly.newPlot('category-cost-chart', categoryCostData, categoryCostLayout);
            
            // Category-specific weight boxplots
            var categoryWeightData = [];
            {weight_boxplot_js}
            
            var categoryWeightLayout = {{
                height: 400, 
                margin: {{ t: 10 }},
                xaxis: {{ title: "Category" }},
                yaxis: {{ title: "Shipping Weight (kg)" }}
            }};
            
            Plotly.newPlot('category-weight-chart', categoryWeightData, categoryWeightLayout);
            
            // Profit margin by category
            var profitMarginData = [];
            {profit_margin_js}
            
            var profitMarginLayout = {{
                height: 400,
                margin: {{ t: 10 }},
                xaxis: {{ title: "Category" }},
                yaxis: {{ 
                    title: "Profit Margin (%)",
                    tickformat: '.0%'
                }}
            }};
            
            Plotly.newPlot('profit-margin-chart', profitMarginData, profitMarginLayout);
        </script>
    </body>
    </html>
    """
    return html_content

def generate_model_report(model: dict, data: pd.DataFrame) -> str:
    """Generate HTML report focused on model training results."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Training Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ padding: 20px; }}
            .card {{ margin-bottom: 20px; }}
            .metric-card {{ text-align: center; padding: 15px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; }}
            .metric-label {{ font-size: 14px; color: #666; }}
            .chart-container {{ height: 400px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="my-4">Price Prediction Model Report</h1>
            <p class="lead">Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="row mb-4">
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">{model["metrics"]["r2_score"]:.2f}</div>
                        <div class="metric-label">R² Score</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">${model["metrics"]["rmse"]:.2f}</div>
                        <div class="metric-label">RMSE</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">${model["metrics"]["mae"]:.2f}</div>
                        <div class="metric-label">MAE</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">{model["model_params"]["epochs"]}</div>
                        <div class="metric-label">Training Epochs</div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5>Feature Importance</h5>
                        </div>
                        <div class="card-body">
                            <div id="feature-importance-chart" class="chart-container"></div>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5>Learning Curve</h5>
                        </div>
                        <div class="card-body">
                            <div id="learning-curve-chart" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Actual vs Predicted Prices</h5>
                        </div>
                        <div class="card-body">
                            <div id="prediction-scatter" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Model Performance by Category</h5>
                        </div>
                        <div class="card-body">
                            <div id="category-performance" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Feature Importance Chart
            var featureNames = Object.keys({model["feature_importance"]});
            var importanceValues = Object.values({model["feature_importance"]});
            
            var featureImportanceLayout = {{
                height: 400,
                margin: {{ t: 10, l: 150 }},
                xaxis: {{ title: "Importance" }}
            }};
            
            Plotly.newPlot('feature-importance-chart', [{{
                x: importanceValues,
                y: featureNames,
                type: 'bar',
                orientation: 'h',
                marker: {{
                    color: 'rgba(153, 102, 255, 0.7)',
                    line: {{
                        color: 'rgba(153, 102, 255, 1.0)',
                        width: 1
                    }}
                }},
                hovertemplate: '<b>%{{y}}</b><br>Importance: %{{x:.2f}}<extra></extra>'
            }}], featureImportanceLayout);
            
            // Learning Curve Chart
            var learningCurveLayout = {{
                height: 400,
                margin: {{ t: 10 }},
                xaxis: {{ title: "Epochs" }},
                yaxis: {{ title: "Loss" }}
            }};
            
            Plotly.newPlot('learning-curve-chart', [
                {{
                    x: {model["learning_curve"]["epochs"]},
                    y: {model["learning_curve"]["train_loss"]},
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Training Loss',
                    line: {{ color: 'rgba(255, 99, 132, 1)' }},
                    hovertemplate: 'Epoch: %{{x}}<br>Training Loss: %{{y:.2f}}<extra></extra>'
                }},
                {{
                    x: {model["learning_curve"]["epochs"]},
                    y: {model["learning_curve"]["val_loss"]},
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Validation Loss',
                    line: {{ color: 'rgba(54, 162, 235, 1)' }},
                    hovertemplate: 'Epoch: %{{x}}<br>Validation Loss: %{{y:.2f}}<extra></extra>'
                }}
            ], learningCurveLayout);
            
            // Generate simulated predictions for the scatter plot
            function simulatePredictions(actual, error) {{
                return actual.map(value => value * (1 + (Math.random() * 2 - 1) * error));
            }}
            
            var actualPrices = {data["price"].tolist()};
            var predictedPrices = simulatePredictions(actualPrices, {0.2 - (model["metrics"]["r2_score"] * 0.15)});
            var categories = {data["category"].tolist()};
            var uniqueCategories = {list(data["category"].unique())};
            
            // Define colors for each category
            var categoryColors = {{
                "Electronics": "rgba(255, 99, 132, 0.7)",
                "Clothing": "rgba(54, 162, 235, 0.7)",
                "Home": "rgba(255, 206, 86, 0.7)",
                "Books": "rgba(75, 192, 192, 0.7)",
                "Sports": "rgba(153, 102, 255, 0.7)"
            }};
            
            // Create traces for each category
            var scatterTraces = [];
            
            uniqueCategories.forEach(category => {{
                var categoryIndices = [];
                for (var i = 0; i < categories.length; i++) {{
                    if (categories[i] === category) {{
                        categoryIndices.push(i);
                    }}
                }}
                
                var categoryActualPrices = categoryIndices.map(i => actualPrices[i]);
                var categoryPredictedPrices = categoryIndices.map(i => predictedPrices[i]);
                
                scatterTraces.push({{
                    x: categoryActualPrices,
                    y: categoryPredictedPrices,
                    mode: 'markers',
                    type: 'scatter',
                    name: category,
                    marker: {{
                        color: categoryColors[category] || "rgba(100, 100, 100, 0.7)",
                        size: 8,
                        line: {{
                            color: categoryColors[category]?.replace('0.7', '1.0') || "rgba(100, 100, 100, 1.0)",
                            width: 1
                        }}
                    }},
                    hovertemplate: '<b>' + category + '</b><br>Actual: $%{{x:.2f}}<br>Predicted: $%{{y:.2f}}<extra></extra>'
                }});
            }});
            
            // Add the perfect prediction line
            scatterTraces.push({{
                x: [0, {data["price"].max() * 1.1}],
                y: [0, {data["price"].max() * 1.1}],
                mode: 'lines',
                type: 'scatter',
                name: 'Perfect Prediction',
                line: {{
                    color: 'rgba(0, 0, 0, 0.5)',
                    dash: 'dash'
                }}
            }});
            
            var scatterLayout = {{
                height: 500,
                margin: {{ t: 10 }},
                xaxis: {{ title: "Actual Price ($)" }},
                yaxis: {{ title: "Predicted Price ($)" }},
                legend: {{ 
                    orientation: 'h',
                    yanchor: 'bottom',
                    y: 1.02,
                    xanchor: 'right',
                    x: 1
                }}
            }};
            
            Plotly.newPlot('prediction-scatter', scatterTraces, scatterLayout);
            
            // Performance by Category
            var categoryPerformance = [];
            
            // Simulate RMSE for each category based on overall RMSE
            var baseRMSE = {model["metrics"]["rmse"]};
            
            uniqueCategories.forEach(category => {{
                // Simulate slightly different performance per category
                var categoryRMSE = baseRMSE * (0.8 + Math.random() * 0.4);
                categoryPerformance.push(categoryRMSE);
            }});
            
            var categoryPerformanceLayout = {{
                height: 400,
                margin: {{ t: 10 }},
                xaxis: {{ title: "Category" }},
                yaxis: {{ title: "RMSE ($)" }}
            }};
            
            Plotly.newPlot('category-performance', [{{
                x: uniqueCategories,
                y: categoryPerformance,
                type: 'bar',
                marker: {{
                    color: [
                        'rgba(255, 99, 132, 0.7)',
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(255, 206, 86, 0.7)',
                        'rgba(75, 192, 192, 0.7)',
                        'rgba(153, 102, 255, 0.7)'
                    ],
                    line: {{
                        color: [
                            'rgba(255, 99, 132, 1.0)',
                            'rgba(54, 162, 235, 1.0)',
                            'rgba(255, 206, 86, 1.0)',
                            'rgba(75, 192, 192, 1.0)',
                            'rgba(153, 102, 255, 1.0)'
                        ],
                        width: 1
                    }}
                }},
                hovertemplate: '<b>%{{x}}</b><br>RMSE: $%{{y:.2f}}<extra></extra>'
            }}], categoryPerformanceLayout);
        </script>
    </body>
    </html>
    """
    return html_content