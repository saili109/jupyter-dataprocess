import pandas as pd
import numpy as np
from ipywidgets import Button, Dropdown,Checkbox, HBox, VBox, Label, Layout, Output, HTML
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
from dataprocess.describe import describe_column, plot_distribution
from dataprocess.null_replace_method import add_scaled, add_normalized, replace_null_with_mean, replace_null_with_median, replace_null_with_knn


CLICK_LAYOUT = Layout(width= '10%', height='25px')
DESCRIBE_LAYOUT = Layout(width= '8%', height='25px')
STATS_LAYOUT = Layout(width='100px', height='25px', margin='0 0 0 0')

# Define dropdown options
null_replace_preprocess_OPTIONS = {
    "is_categorical": ['Categorical', "Discrete", 'Continuous', 'Null'], 
    "null_replace_options": [[""],["None", "Median", "KNN"], ["None", "Mean", "Median"], ['None', 'Drop']]
}

# Define constants (a line, titles) in figure and statistics SHOW
SEPARATOR_LINE = HTML('<hr style="margin: 0; border: 1px solid #9A9696"/>', layout=Layout(width='500px', ))
# make the title for statistics table
minimum_title = Label(value='Minimum', layout=STATS_LAYOUT)
maximum_title = Label(value='Maximum', layout=STATS_LAYOUT)
median_title = Label(value='Median', layout=STATS_LAYOUT)
mean_title = Label(value='Mean', layout=STATS_LAYOUT)
std_title = Label(value='SD', layout=STATS_LAYOUT)
STATISTICS_TITLE_hbox = HBox([minimum_title, maximum_title, median_title, mean_title, std_title])

class WidgetRow(object):
    """ Fill in each Widget row with column information"""

    def __init__(self, dataframe, column):
        self.dataframe = dataframe
        self.column = column

        # Record choices for processing steps
        self.categorical_value = None
        self.null_replacement_choice = None
        self.preprocess_choice = None
        self.show_original_figure = False
        self.show_new_figure = False
        
        ## Description section ##
        
        # Place datatype and null_percentage information
        self.colname_widget = Label(value=self.column.name, layout=Layout(width= '15%', height='25px')) ###### adjust to make it look nicer
        self.dtype_widget = Label(layout=DESCRIBE_LAYOUT)
        self.null_percentage_widget = Label(layout=DESCRIBE_LAYOUT)
        self.distinct_widget =  Label(layout=DESCRIBE_LAYOUT)
        self.categorical_dropdown = Dropdown(
            options=['Categorical',"Discrete", 'Continuous', 'Null'],
            value='Continuous',
            disabled=False,
            layout=CLICK_LAYOUT
            )
        
        # Place distribution figures
        self.fig_old_widget_wrapper = VBox([])
        self.fig_new_widget_wrapper = VBox([])
        
        # Place original statistcs (mean, median etc)
        original_description = describe_column(self.column)
        self.original_statistics = HBox([
            Label(value=original_description['minimum'], layout=STATS_LAYOUT),
            Label(value=original_description['maximum'], layout=STATS_LAYOUT),
            Label(value=original_description['median'], layout=STATS_LAYOUT),
            Label(value=original_description['mean'], layout=STATS_LAYOUT),
            Label(value=original_description['std'], layout=STATS_LAYOUT)
            ])
            
        # Place new statistics
        (self.minimum_widget, 
        self.maximum_widget, 
        self.median_widget,
        self.mean_widget,
        self.std_widget) = [Label(layout=STATS_LAYOUT) for i in range(5)]
        
        # Define preprocessing dropdown
        self.knn_checkbox_widget = HBox([])  
        self.null_replacement_dropdown = Dropdown(
        options=['None', 'Mean', 'Median', "KNN"],
        value='None',
        disabled=False if self.column.isnull().sum() > 0 else True,
        layout=CLICK_LAYOUT
        )
        
        self.preprocess_dropdown = Dropdown(
            options=['None', 'Scale', 'Normalize'],
            value='None',
            disabled=False if self.column.isnull().sum() == 0 else True,
            layout=CLICK_LAYOUT
            )
            
    ## UI for KNN replacement       
    def create_knn_checkbox(self):
        """ Make a checkbox for KNN model predictors selection"""
        
        knn_choice = []
        for colname in self.dataframe.columns:
            if (self.dataframe[colname].dtype in ['int64', 'float64'] 
            and self.dataframe[colname].isnull().sum()/float(len(self.dataframe[colname])) < 1):
                knn_choice.append(colname)

        checkbox_list = []
        for colname in knn_choice:
            checkbox = Checkbox(
                value=True,
                description=colname,
                disabled=False,
                indent=False
            )
            checkbox_list.append(checkbox)
        return checkbox_list
    
    def null_knn_replacement_calculation(self):
        """Replace with KNN and populate new values"""
        
        knn_checkbox_instr = Label(value='Please select predictors for the KNN model: ', layout=Layout(width='40%', height='25px'))
        knn_submit_button = Button(description='Submit')
        knn_checkbox = self.create_knn_checkbox()
        self.knn_checkbox_widget.children = [knn_checkbox_instr, VBox(knn_checkbox), knn_submit_button]
        
        # populate the original information before the knn replacement is done
        self.populate(self.column) 
        
        def click_knn_submit_button(x):
            checked_columns = []
            for i in range(len(knn_checkbox)):
                if knn_checkbox[i].value:
                    checked_columns.append(knn_checkbox[i].description)
            self.knn_checkbox_widget.children = []
            print('Predictors: ', checked_columns)
            column = replace_null_with_knn(self.dataframe, self.column.name, checked_columns)
            self.dataframe[self.column.name+"_knn"] = column # put it in a new dataframe
            self.preprocess_dropdown.disabled = False
            
            self.populate(column)
        knn_submit_button.on_click(click_knn_submit_button)    
    
    ## Populate information and update the column        
    def populate(self, column):
        """ Input the description, statistics and figure of the column """ 
        
        description = describe_column(column)
        
        self.dtype_widget.value = description['dtype']
        self.null_percentage_widget.value = description['null_percentage']
        self.distinct_widget.value = description['distinct']
        if self.categorical_value is None:
            
            self.categorical_dropdown.value = description['categorical']
        else:
            self.categorical_dropdown.value = self.categorical_value
        
        # Input options for the dropdown based on is_categorical
        options = pd.DataFrame(null_replace_preprocess_OPTIONS)
        self.null_replacement_dropdown.options = options.loc[options['is_categorical'] == self.categorical_dropdown.value, "null_replace_options"].iloc[0]
       
        # Input statistics and figure for the SHOW click 
        self.minimum_widget.value = description['minimum']
        self.maximum_widget.value = description['maximum']
        self.median_widget.value = description['median']
        self.mean_widget.value = description['mean']
        self.std_widget.value = description['std']
        populated_statistics = HBox([self.minimum_widget, self.maximum_widget, self.median_widget, self.mean_widget, self.std_widget])

        # Define UI to show figures
        if self.show_original_figure:
            # statistics table
            statistics_vbox = VBox([
            SEPARATOR_LINE,
            HBox(
            [Label(value='Original', layout=Layout(margin='0 auto 0 auto'), size = 25)], 
            layout=Layout(width='500px', margin='0 0 0 0')),
            STATISTICS_TITLE_hbox,
            self.original_statistics
            ]) 
            # plot
            output_plot = Output(layout=Layout(width='500px'))
            with output_plot:
                plot_distribution(column=self.column, title='', is_categorical=self.categorical_value)
                display(plt.gcf())
            self.fig_old_widget_wrapper.children = [statistics_vbox, output_plot, SEPARATOR_LINE]
        else:
            self.fig_old_widget_wrapper.children = []

        if self.show_new_figure:
            # statistics table
            statistics_vbox = VBox([
            SEPARATOR_LINE,
            HBox(
            [Label(value='New', layout=Layout(margin='0 auto 0 auto'), size = 25)], 
            layout=Layout(width='500px', margin='0 0 0 0')),
            STATISTICS_TITLE_hbox,
            populated_statistics
            ]) 
            # plot
            output_plot = Output(layout=Layout(width='500px'))
            with output_plot:
                plot_distribution(column=column, title='', is_categorical=self.categorical_value)
                display(plt.gcf())
            self.fig_new_widget_wrapper.children = [statistics_vbox, output_plot, SEPARATOR_LINE]
        else:
            self.fig_new_widget_wrapper.children = []
            
    def compute_processed_column(self):
        column = self.column
        
        if self.null_replacement_choice == 'mean':
            column = replace_null_with_mean(column)
            
        elif self.null_replacement_choice == 'median':
            column = replace_null_with_median(column)
        elif self.null_replacement_choice == 'drop':
            pass
        elif self.null_replacement_choice == 'knn' and self.column.name+'_knn' in self.dataframe.columns:
            column = self.dataframe[self.column.name+"_knn"]
    
        if self.preprocess_choice == 'scale':
            column = add_scaled(column)
        elif self.preprocess_choice == 'norm':
            column = add_normalized(column)
        
        return column
            
    def _update(self):
        """Rerun the processing pipeline and return the processed column."""
        column = self.compute_processed_column()
        self.populate(column)

    def create_figshow_checkbox(self, choice):
        """Create the checkbox to show figure"""
        checkbox = Checkbox(
            value=False,
            description=choice,
            disabled=False,
            indent=False,
            layout=Layout(width='80px', height='25px')
            )
        return checkbox

    def create_h_widget(self):
        fig_original_checkbox = self.create_figshow_checkbox('Original')
        fig_new_checkbox = self.create_figshow_checkbox('New')
        fig_hbox = HBox([self.fig_old_widget_wrapper, self.fig_new_widget_wrapper])
        knn_checkbox_widget = self.knn_checkbox_widget

        descripion_hbox = HBox([
            self.colname_widget,
            self.dtype_widget,
            self.null_percentage_widget,
            self.distinct_widget,
            self.categorical_dropdown,
            
            self.null_replacement_dropdown,
            self.preprocess_dropdown,
            fig_original_checkbox,
            fig_new_checkbox
        ])
        
        col_widget = VBox([descripion_hbox, knn_checkbox_widget, fig_hbox])
        
        def on_categorical_dropdown_change(change):
            self.categorical_value = self.categorical_dropdown.value
            self._update()
        self.categorical_dropdown.observe(on_categorical_dropdown_change, names='value')
            

        def on_null_replacement_dropdown_change(change):
            mapping = {
                'Mean': 'mean',
                'Median': 'median',
                'Drop': 'drop',
                'KNN': 'knn'
            }
            self.null_replacement_choice = mapping.get(
                self.null_replacement_dropdown.value
            )
            
            if self.null_replacement_choice == 'None' or self.null_replacement_choice == None:
                self.preprocess_dropdown.value = 'None'
                self.preprocess_choice = None
                self.preprocess_dropdown.disabled = True
                self.knn_checkbox_widget.children = []
                self._update()
                
            elif self.null_replacement_choice in ('mean', 'median'):
                self._update()
                self.preprocess_dropdown.disabled = False
                self.knn_checkbox_widget.children = []
            elif self.null_replacement_choice == 'drop':
                pass
            elif self.null_replacement_choice == 'knn':
                if self.column.name+'_knn' in self.dataframe.columns:
                     self._update()
                     self.preprocess_dropdown.disabled = False
                else:
                    self.preprocess_dropdown.value = 'None'
                    self.preprocess_choice = None
                    self.preprocess_dropdown.disabled = True
                    
                    fig_original_checkbox.value = False
                    fig_new_checkbox.value = False
                    self.null_knn_replacement_calculation()
        self.null_replacement_dropdown.observe(on_null_replacement_dropdown_change, names='value')
        
        def on_preprocess_dropdown_change(change):
            mapping = {
                'Scale': 'scale',
                'Normalize': 'norm'
            }
            self.preprocess_choice = mapping.get(self.preprocess_dropdown.value)
            self._update()
        self.preprocess_dropdown.observe(on_preprocess_dropdown_change, names='value')

        def on_fig_original_checkbox_value_change(change):
            self.show_original_figure = fig_original_checkbox.value
            self._update()
        fig_original_checkbox.observe(on_fig_original_checkbox_value_change, names='value')

        def on_fig_new_checkbox_value_change(change):
            self.show_new_figure = fig_new_checkbox.value
            self._update()
        fig_new_checkbox.observe(on_fig_new_checkbox_value_change, names='value')

        return col_widget

    def get_h_widget(self):
        self._update()
        return self.create_h_widget()
        
    

class CreateWidgets(object):
    """Show the summary and button the replace nulls for a single column """

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.colnames = dataframe.columns
        self.tablewidget = VBox(self.create_h_widgets())

    def create_h_widgets(self):
        """"Create the widget table"""
        dataframe = self.dataframe

        colname_title = Label(value='NAME', layout=Layout(width= '15%', height='25px'))
        dtype_title = Label(value='TYPE', layout=DESCRIBE_LAYOUT)
        null_percentage_title = Label(value='NULL', layout = DESCRIBE_LAYOUT)
        distinct_title = Label(value='DISTINCT', layout = DESCRIBE_LAYOUT)
        
        categorical_title = Label(value='IS CATEGORICAL', layout=CLICK_LAYOUT)
        algorithm_choice_title = Label(value='NULL REPLACE', layout=CLICK_LAYOUT)
        preprocess_title = Label(value='PREPROCESS', layout=CLICK_LAYOUT)
        show_title = Label(value='SHOW', layout=CLICK_LAYOUT)
        header_hbox = HBox([
            colname_title,
            dtype_title, 
            null_percentage_title, 
            distinct_title, 
            categorical_title, 
            algorithm_choice_title, 
            preprocess_title, 
            show_title
        ])
      
        # create each widget row
        self.rows = []
        for colname in self.colnames:
            row = WidgetRow(self.dataframe, self.dataframe[colname])
            self.rows.append(row)
        content_rows = [row.get_h_widget() for row in self.rows]
        
        return [header_hbox] + content_rows
        
    def processed_dataframe(self):
        """Generate processed dataframe"""
        processed_columns = {}
        for row in self.rows:
            name = row.column.name
            processed_column = row.compute_processed_column()
            processed_columns[name] = processed_column
        return pd.DataFrame(processed_columns)

    def get_widget(self):
        return self.tablewidget
