import matplotlib.pyplot as plt
import seaborn as sns
import mpld3
import pandas as pd
import numpy as np
  
# import file with data
x_test  = np.load('../../data/test_monthmean.npy')
x_test = x_test.reshape((x_test.shape[0],-1,x_test.shape[-1]))
x_test = np.delete(x_test,(57,58,59),axis = 1)
x_test = x_test.reshape((-1,38))
df = pd.DataFrame(x_test)
df.columns = ['lat 0', 'lon 1', 'time 2', 'agb 3', 'pft_fracCover 4', 'sm 5', 'pftCrop 6',
       'pftHerb 7', 'pftShrubBD 8', 'pftShrubNE 9', 'pftTreeBD 10', 'pftTreeBE 11',
       'pftTreeND 12', 'pftTreeNE 13', 'GDP 14', 'ign 15', 'Distance_to_populated_areas 16',
       'fPAR 17', 'LAI 18', 'NLDI 19', 'vod_K_anomalies 20', 'FPAR_12mon 21', 'LAI_12mon 22',
       'Vod_k_anomaly_12mon 23', 'FPAR_06mon 24', 'LAI_06mon 25', 'Vod_k_anomaly_06mon 26',
       'WDPA_fracCover 27', 'dtr 28', 'pet 29', 'tmx 30', 'wet 31', 'Biome 32', 'precip 33',
       'Livestock 34', 'road_density 35', 'topo 36', 'pop_density 37']


# Calculate correlations
corr = df.corr()

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(10,10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, annot=False, fmt='.2f', ax=ax)

# Add tooltips to the heatmap cells
mpld3.plugins.connect(plt.gcf(), mpld3.plugins.MousePosition(fontsize=14))

# Save the plot as an HTML file
mpld3.save_html(plt.gcf(), 'heatmap.html')

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
# # import file with data
# x_test  = np.load('../../data/test_monthmean.npy')
# x_test = x_test.reshape((x_test.shape[0],-1,x_test.shape[-1]))
# x_test = np.delete(x_test,(57,58,59),axis = 1)
# x_test = x_test.reshape((-1,38))
# df = pd.DataFrame(x_test)
# df.columns = ['lat 0', 'lon 1', 'time 2', 'agb 3', 'pft_fracCover 4', 'sm 5', 'pftCrop 6',
#        'pftHerb 7', 'pftShrubBD 8', 'pftShrubNE 9', 'pftTreeBD 10', 'pftTreeBE 11',
#        'pftTreeND 12', 'pftTreeNE 13', 'GDP 14', 'ign 15', 'Distance_to_populated_areas 16',
#        'fPAR 17', 'LAI 18', 'NLDI 19', 'vod_K_anomalies 20', 'FPAR_12mon 21', 'LAI_12mon 22',
#        'Vod_k_anomaly_12mon 23', 'FPAR_06mon 24', 'LAI_06mon 25', 'Vod_k_anomaly_06mon 26',
#        'WDPA_fracCover 27', 'dtr 28', 'pet 29', 'tmx 30', 'wet 31', 'Biome 32', 'precip 33',
#        'Livestock 34', 'road_density 35', 'topo 36', 'pop_density 37']


# # Calculate correlations
# corr = df.corr()


# # Step 1. Create a heatmap

# fig, ax = plt.subplots(figsize=(10,10))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(220, 10, as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
# heatmap = sns.heatmap(corr, cmap=cmap, annot=False, fmt='.2f', ax=ax)

# # Step 2. Create Annotation Object
# annotation = ax.annotate(
#     text='',
#     xy=(0, 0),
#     xytext=(15, 15), # distance from x, y
#     textcoords='offset points',
#     bbox={'boxstyle': 'round', 'fc': 'w'},
#     arrowprops={'arrowstyle': '->'}
# )
# annotation.set_visible(False)


# # Step 3. Implement the hover event to display annotations
# def motion_hover(event):
#     annotation_visbility = annotation.get_visible()
#     if event.inaxes == ax:
#         is_contained, annotation_index = heatmap.contains(event)
#         if is_contained:
#             data_point_location = heatmap.get_offsets()[annotation_index['ind'][0]]
#             annotation.xy = data_point_location

#             text_label = '({0:.2f}, {0:.2f})'.format(data_point_location[0], data_point_location[1])
#             annotation.set_text(text_label)

#             annotation.get_bbox_patch().set_facecolor(cmap(norm(colors[annotation_index['ind'][0]])))
#             annotation.set_alpha(0.4)

#             annotation.set_visible(True)
#             fig.canvas.draw_idle()
#         else:
#             if annotation_visbility:
#                 annotation.set_visible(False)
#                 fig.canvas.draw_idle()

# fig.canvas.mpl_connect('motion_notify_event', motion_hover)

# plt.show()