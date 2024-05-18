from scipy.spatial.transform import Rotation as R

r1 = R.from_euler('XYZ', [0, 0, 45], degrees=True)
r2 = R.from_euler('XYZ', [1, 23, 34], degrees=True)

r = r1 * r2
print(r.as_euler('XYZ', degrees=True))
