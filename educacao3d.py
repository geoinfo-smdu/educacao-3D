import json
import geopandas as gpd
import pandas as pd
from shapely import geometry, wkt
import pdal
from sklearn.cluster import DBSCAN #, OPTICS
from sklearn import preprocessing
import numpy as np
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
import math

def get_points(lote):
    s = lote
    bounds = ([s.bounds[0], s.bounds[2]], [s.bounds[1], s.bounds[3]])

    ept = {
        "pipeline":[
            {
            "type": "readers.ept",
            "filename": "https://ept-m3dc-pmsp.s3-sa-east-1.amazonaws.com/ept.json",
            "bounds": str(bounds)
            },
            {
                "type":"filters.crop",
                "polygon":s.wkt
            },
            {   
                "type":"filters.hag_delaunay"
            }
        ]}

    pipeline = pdal.Pipeline(json.dumps(ept))
    pipeline.validate()
    n_points = pipeline.execute()

    arr = pipeline.arrays[0]
    df = pd.DataFrame(arr)

    return df

def voxelize(df):
    # Reduzindo valores a mínimos para poderem ser visualizados
    coord_minimas = df[['X', 'Y', 'Z']].min()
    df[['X', 'Y', 'Z']] = df[['X', 'Y', 'Z']] - coord_minimas

    ## Separando apenas Buildings
    bd = df[df.Classification.isin([6])].reset_index()

    bd.index.name = 'id'

    CELL = 0.40

    bd.X, bd.Y = round(round((1/CELL) * bd.X) * CELL, 1), round(round((1/CELL) * bd.Y) * CELL, 1)

    z_max = bd[['X', 'Y', 'Z', 'HeightAboveGround']].groupby(['X', 'Y']).max()

    bd.set_index(['X', 'Y']).Z = z_max

    return pd.DataFrame(z_max, columns=['Z', 'HeightAboveGround']).reset_index()

def clustering(bd_voxel):
    X = bd_voxel[['X', 'Y', 'Z']]

    EPS = 1.5
    MIN_SAMPLES = 6

    clustering = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(X)

    bd_voxel['ClusterID'] = clustering.labels_

    return bd_voxel

def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    triangles = coords[tri.vertices]
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - triangles[:,0,1]) ** 2) ** 0.5
    s = ( a + b + c ) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:,(0,1)]
    edge2 = filtered[:,(1,2)]
    edge3 = filtered[:,(2,0)]
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
#     return cascaded_union(triangles[0])
    return cascaded_union(list(filter(lambda x: x.buffer(-0.20).intersection(points).type == 'MultiPoint', triangles)))
