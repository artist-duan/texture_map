import numpy as np
from tqdm import tqdm


def heron_formula(us, vs):
    a = ((us[0] - us[1]) ** 2 + (vs[0] - vs[1]) ** 2) ** (1 / 2)
    b = ((us[0] - us[2]) ** 2 + (vs[0] - vs[2]) ** 2) ** (1 / 2)
    c = ((us[1] - us[2]) ** 2 + (vs[1] - vs[2]) ** 2) ** (1 / 2)
    p = (a + b + c) / 2
    s = (p * (p - a) * (p - b) * (p - c)) ** (1 / 2)
    return s


def softmax(data, coefficient=1.0):
    data -= np.max(data)
    data = np.exp(coefficient * data)
    data = data / np.sum(data)
    return data


def KLDivergence(p, q):
    return -np.sum(p * np.log(q)) + np.sum(p * np.log(p))


def gen_affinity(
    n_images,
    triangles,
    vertices,
    visibles,
    face_adjacencys,
    softmax_coefficient=1.0,
    adjacency_level=1,
    sample_num=40,
):
    tqdm.write("Affinity >>>>>>>>>>>>>>>>")

    probs, areas = [], []
    for i in tqdm(range(len(visibles)), total=len(visibles)):
        visible = visibles[i]
        prob = np.zeros((n_images,), dtype=np.float32)
        if not visible:
            areas.append(prob)
            prob = softmax(prob, coefficient=softmax_coefficient)
            probs.append(prob)
            continue

        for vis in visible:
            index, u, v, angle, d, d_ = vis
            prob[index] = heron_formula(u, v)
        areas.append(prob)
        prob = softmax(prob, coefficient=softmax_coefficient)
        probs.append(prob)

    rows, cols, labels, distances = [], [], [], []
    for i in tqdm(range(len(visibles)), total=len(visibles)):
        adjacencys = []
        fas = face_adjacencys[i]
        adjacencys += fas
        for j in range(adjacency_level):
            tmp = []
            for fa in fas:
                tmp += face_adjacencys[fa]
            adjacencys += tmp
            fas = tmp
        p1, p2, p3 = triangles[i]
        c = (vertices[p1] + vertices[p3] + vertices[p2]) / 3.0
        adjacencys = list(set(adjacencys))
        if len(adjacencys) > sample_num:
            adjacencys = np.random.choice(adjacencys, sample_num)

        for adj in adjacencys:
            p1, p2, p3 = triangles[adj]
            c1 = (vertices[p1] + vertices[p3] + vertices[p2]) / 3.0
            dis = (((c - c1) ** 2).sum()) ** (1 / 2)

            rows.append(i)
            cols.append(adj)
            labels.append(KLDivergence(probs[i], probs[adj]))
            distances.append(dis)

    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    labels = np.array(labels, dtype=np.float32)
    distances = np.array(distances, dtype=np.float32)
    areas = np.array(areas, dtype=np.float32)
    probs = np.array(probs, dtype=np.float32)
    return rows, cols, labels, distances, probs, areas
