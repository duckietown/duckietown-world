import plotly.offline as offline

res = offline.plot({'data': [{'y': [4, 2, 3, 4]}],
                    'layout': {'title': 'Test Plot',
                               'font': dict(size=16)}},
                   output_type='div')

with open('testplotly.html', 'w') as f:
    f.write('<div style="width: 600px; height: 300">')
    f.write(res)
    f.write('</div>')


