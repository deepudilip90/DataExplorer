import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


class StructuredDataExplorer:

    def __init__(self, data_path, target_type, target=None, output_path=None):
        self.df = self.read_data(data_path)
        self.target_type = target_type
        self.target_col = target
        self.output_path = output_path
        self.numeric_composition = None
        self.categorical_composition = None
        self.number_of_categories = None

        if self.target_col is None:
            print('Target Column not specified. Please create manuallly')

        self.rows = self.df.shape[0]
        self.columns = self.df.shape[1]
        self.numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        self.object_cols = self.df.select_dtypes(include='object').columns.tolist()
        self.category_cols = self.df.select_dtypes(include='category').columns.tolist()
        self.bool_cols = self.df.select_dtypes(include='bool').columns.tolist()
        #         self.date_cols = self._detect_date_cols()
        self.uncategorized_data_cols = [col for col in self.df.columns
                                        if col not in
                                        self.df.select_dtypes(include=['number', 'object',
                                                                       'category', 'datetime',
                                                                       'bool'])]
        self._get_attr_categorical_composition()

    def read_data(self, data_path):

        print(f'Reading data from {data_path}')

        if data_path.endswith('csv'):
            data = pd.read_csv(data_path)

        elif data_path.endswith('xlsx'):
            data = pd.read_excel('xlsx')
        return data

    #     def _get_object_cols:
    #         pass

    #     def _get_category_cols:
    #         pass

    #     def _get_numeric_cols:
    #         pass

    #     def _get_bool_cols:
    #         pass

    #     def _get_date_cols:
    #         pass

    def show_data_info(self):
        print(self.df.info())

    #     def _detect_date_cols(self):
    #         print('checking and creating date type columns')
    #         date_cols = []
    #         for col in self.object_cols:
    #             try:
    #                 self.df[col] = pd.to_datetime(self.df[col])
    #                 date_cols.append(col)
    #             except ValueError:
    #                 pass
    #         date_col_string = ' '.join([str(x) for x in date_cols])
    #         print(f"columns {date_col_string} converted to type datetime")
    #         return date_cols

    def get_fill_rate(self):
        self.fillrate_columns = (self.df.count() / self.rows).to_dict()
        print(self.fillrate_columns)

    def remove_bad_cols(self, min_fill_rate):
        bad_cols = [k for k, v in self.fillrate_columns.items() if v < min_fillrate]
        self.df = self.df[[col for col in self.df.columns if col not in bad_cols]]

    def _get_categories(self, column):
        return self.df[column].unique().tolist()

    def _filter_category(self, col, categories):
        self.df = self.df[self.df[col].isin(categories)]

    def _get_ranges(self, column):
        return [np.min(self.df[column]), np.max(self.df[column])]

    def _get_attr_categorical_composition(self, max_categories=20):
        category_dict = {}
        category_number_dict = {}
        for col in self.category_cols + self.object_cols:
            categories = self._get_categories(col)
            category_number_dict[col] = len(categories)

            if len(categories) > max_categories:
                continue
            else:
                category_dict[col] = self._get_categories(col)

        self.categorical_composition = category_dict
        self.number_of_categories = category_number_dict

    def _get_attr_numeric_composition(self):

        numeric_dict = {}
        for col in self.numeric_cols:
            numeric_dict[col] = self._get_ranges(col)

        self.numeric_composition = numeric_dict

    def get_data_composition(self, max_categories=20):

        if self.categorical_composition is None:
            self._get_attr_categorical_composition(max_categories)

        elif max_categories != len(self.categorical_composition):
            self._get_attr_categorical_composition(max_categories)

        if self.numeric_composition is None:
            self._get_attr_numeric_composition()

        print('Ignoring following column that exceed maximum number of categories specified')
        ignore_cols = [col for col in self.category_cols + self.object_cols
                       if col not in self.categorical_composition.keys()]
        print(*ignore_cols, sep='\n')

        return self.categorical_composition, self.numeric_composition

    @staticmethod
    def _get_next_plot_index(i, j, i_max, j_max):
        j = j + 1
        if j >= j_max:
            i = i + 1
            j = 0
            if i >= i_max:
                return 0, 0
        return i, j

    @staticmethod
    def _set_plot_attributes_category(ax):
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    def _create_bar_plot(self, ax, x_col, y_col=None, add_perc=True, data=None):

        if data is None:
            data = self.df

        if y_col is not None:
            sns.barplot(x=x_col, y=y_col, data=data, ax=ax, orient='v')
        else:
            sns.barplot(x=x_col, y=x_col, data=data, estimator=lambda x: len(x), orient='v', ax=ax)

        if add_perc:
            for p in ax.patches:

                if y_col is not None:
                    total = sum(data[y_col])
                else:
                    total = len(data[x_col])

                percentage = '{:.1f}%'.format(100 * p.get_height() / total)
                x = p.get_x() + p.get_width() / 2
                y = p.get_y() + p.get_height() + 2
                ax.annotate(percentage, (x, y))

    def _create_histogram(self, ax, col, kde=False, data=None):

        if data is None:
            data = self.df

        sns.distplot(self.df[col], ax=ax, kde=kde)

    def _create_boxplot(self, ax, col, data=None):

        if data is None:
            data = self.df

        sns.boxplot(y=col, data=self.df, ax=ax)

    def _create_oneway_plot(self, ax, col, data=None):

        if data is None:
            data = self.df

        if self.target_type == 'continuous':
            df = self.df.groupby(col, as_index=False).agg({self.target_col: ['mean', 'count']})
            df['category_perc'] = df[(self.target_col, 'count')] / sum(df[(self.target_col, 'count')])
        else:
            df = self.df.groupby(col, as_index=False).agg({self.target_col: ['sum', 'count']})
            df['response_rate'] = df[(self.target_col, 'sum')] / sum(df[(self.target_col, 'count')])
            df['category_perc'] = df[(self.target_col, 'count')] / sum(df[(self.target_col, 'count')])

        #         sns.barplot(x=col, y='category_perc', data=df, ax=ax)
        print(df.columns)
        self._create_bar_plot(ax=ax, x_col=col, y_col=(self.target_col, 'count'), data=df)

        ax2 = ax.twinx()
        ax2.grid(False)

        if self.target_type == 'continuous':
            sns.pointplot(x=col, y=(self.target_col, 'mean'), data=df, ax=ax2, ci=None)
        else:
            sns.pointplot(x=col, y='response_rate', data=df, ax=ax2, ci=None)

    def _create_set_of_plots(self, plot_cols, plot_type, fig_name, max_plots_x=3, max_plots_y=3,
                             alter_attributes=True):
        plot_x = 0
        plot_y = 0
        figure_number = 0

        for col in plot_cols:
            if (plot_x == 0) & (plot_y == 0):
                if figure_number > 0:
                    fig.savefig(fig_name + '_' + str(figure_number) + '.png')

                fig, ax = plt.subplots(max_plots_x, max_plots_y, figsize=(20, 15))
                fig.tight_layout(h_pad=5)
                figure_number += 1

            print('now adding plot for: ', col)
            if alter_attributes:
                self._set_plot_attributes_category(ax[plot_x, plot_y])

            if plot_type == 'bar_of_count':
                self._create_bar_plot(ax=ax[plot_x, plot_y], x_col=col)

            elif plot_type == 'histogram':
                self._create_histogram(ax=ax[plot_x, plot_y], col=col)

            elif plot_type == 'boxplot':
                self._create_boxplot(ax=ax[plot_x, plot_y], col=col)

            elif plot_type == 'oneway_plot':
                self._create_oneway_plot(ax=ax[plot_x, plot_y], col=col)

            plot_x, plot_y = self._get_next_plot_index(plot_x, plot_y, max_plots_x, max_plots_y)

        fig.savefig(fig_name + '_' + str(figure_number) + '.png', bbox_inches='tight')

    def get_data_distribution_categorical(self, max_plots_x=3, max_plots_y=3, max_categories=20):

        plot_cols = [col for col in self.category_cols + self.object_cols
                     if len(self._get_categories(col)) < max_categories]

        plot_cols = plot_cols + self.bool_cols

        print('plotting only for columns: ', plot_cols)

        self._create_set_of_plots(plot_cols=plot_cols, plot_type='bar_of_count',
                                  fig_name='distribution_categorical')

    def get_data_distribution_numeric(self, max_plots_x=3, max_plots_y=3, max_categories=20):

        plot_cols = self.numeric_cols

        print('plotting for columns: ', plot_cols)

        self._create_set_of_plots(plot_cols=plot_cols, plot_type='histogram',
                                  fig_name='histogram_numeric', alter_attributes=False)
        self._create_set_of_plots(plot_cols=plot_cols, plot_type='boxplot',
                                  fig_name='boxplot_numeric', alter_attributes=False)

    def create_bins(self, in_col, out_col=None, no_of_bins=10):

        if out_col is None:
            out_col = in_col + '_bin'
        self.df[out_col] = pd.cut(self.df[in_col], bins=no_of_bins)

    def get_oneway_plots(self, max_categories=20):

        # create bins for numeric cols if not present
        for col in self.numeric_cols:
            if col + '_bin' not in self.df.columns:
                self.create_bins(col)

        plot_cols_categorical = [col for col in self.category_cols + self.object_cols
                                 if len(self._get_categories(col)) < max_categories]

        plot_cols_numeric = [col for col in self.df.columns if '_bin' in col]

        plot_cols = plot_cols_categorical + plot_cols_numeric + self.bool_cols

        print('plotting only for columns: ', plot_cols)

        self._create_set_of_plots(plot_cols=plot_cols, plot_type='oneway_plot',
                                  fig_name='oneway_plot', max_plots_x=2, max_plots_y=2)

    def chisq_test(self, col_1, col_2, alpha):
        cont_table = pd.crosstab(self.df[col_1], self.df[col_2])
        stat, p, dof, expected = chi2_contingency(table)
        print('test_statistic: ', stat)
        print('p - value: ', p)
        print('degrees of freedom: ', dof)

        if p < alpha:
            print('Reject Null: The provided categorical columns are dependent')
        else:
            print('Fail to reject Null: The provided categorical columns are independent')


