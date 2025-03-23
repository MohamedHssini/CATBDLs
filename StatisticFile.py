from xhtml2pdf import pisa


class StatisticTable:
    def __init__(self, test_set_data, type_generator_index, file_name="Test_Accuracy_Statistic.pdf"):
        self.test_set_data = test_set_data
        self.file_name = file_name
        self.type_generator_index = type_generator_index

    def generate_table_pdf(self):
        html_content = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Statistic Table</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                }
                .table-container {
                    width: 500px;  /* Fixed container width */
                    margin: auto;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th, td {
                    border: 1px solid black;
                    padding: 4px; /* Reduce padding */
                    font-size: 9px; /* Reduce font size */
                    text-align: center;
                }
                th {
                    background-color: #f2f2f2;
                }
                .section-header {
                    background-color: #d9edf7;
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
        <div class="table-container">
        <table>
            <tr>
                <th rowspan="2">Test set</th>
        '''

        # Add columns based on the type_generator_index
        if self.type_generator_index == 0:
            html_content += '''
                <th colspan="3">Yager t-norms</th>
                <th colspan="3">Aczel-Alsina t-norms</th>
            '''
        elif self.type_generator_index == 1:
            html_content += '''
                <th colspan="3">Yager t-norms</th>
            '''
        elif self.type_generator_index == 2:
            html_content += '''
                <th colspan="3">Aczel-Alsina t-norms</th>
            '''

        html_content += '''
            </tr>
            <tr>
        '''

        # Add the specific columns based on the type_generator_index
        if self.type_generator_index == 0:
            html_content += '''
                <th>λ</th><th>Av Acc</th><th>Stddev</th>
                <th>λ</th><th>Av Acc</th><th>Stddev</th>
            '''
        elif self.type_generator_index == 1:
            html_content += '''
                <th>λ</th><th>Av Acc</th><th>Stddev</th>
            '''
        elif self.type_generator_index == 2:
            html_content += '''
                <th>λ</th><th>Av Acc</th><th>Stddev</th>
            '''

        html_content += '''</tr>'''

        # Loop through test sets and generate table rows dynamically
        for test_set in self.test_set_data:
            if self.type_generator_index == 0:
                html_content += f'<tr class="section-header"><td colspan="7">{test_set["test_set"]}%</td></tr>'
            else:
                html_content += f'<tr class="section-header"><td colspan="4">{test_set["test_set"]}%</td></tr>'
            first_row = True  # Track first row to merge "Test set" column

            for row in test_set["rows"]:
                if first_row:
                    html_content += f'<tr><td rowspan="{len(test_set["rows"])}">{test_set["test_set"]}%</td>'
                    first_row = False
                else:
                    html_content += '<tr>'

                # Add data based on type_generator_index
                if self.type_generator_index == 0:
                    html_content += f'''
                        <td>{row["λ_Yager"]}</td><td>{row["Av Acc_Yager"] if row["Av Acc_Yager"] == "-" else f'{row["Av Acc_Yager"]}%'}</td><td>{row["Stddev_Yager"]}</td>
                        <td>{row["λ_Acal"]}</td><td>{row["Av Acc_Acal"] if row["Av Acc_Acal"] == "-" else f'{row["Av Acc_Acal"]}%'}</td><td>{row["Stddev_Acal"]}</td>
                    '''
                elif self.type_generator_index == 1:
                    html_content += f'''
                        <td>{row["λ"]}</td><td>{row["Av Acc"] if row["Av Acc"] == "-" else f'{row["Av Acc"]}%'}</td><td>{row["Stddev"]}</td>
                    '''
                elif self.type_generator_index == 2:
                    html_content += f'''
                        <td>{row["λ"]}</td><td>{row["Av Acc"] if row["Av Acc"] == "-" else f'{row["Av Acc"]}%'}</td><td>{row["Stddev"]}</td>
                    '''
                html_content += '</tr>'

        # Close table and HTML
        html_content += '''
        </table>
        </div>
        </body>
        </html>
        '''

        # Convert HTML to PDF using xhtml2pdf
        with open(self.file_name, "w+b") as pdf_file:
            pisa.CreatePDF(html_content, dest=pdf_file, encoding='UTF-8')
