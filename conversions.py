import numpy as np
import cv2
import warnings

def non_linear_rgb_to_linear_rgb(image):
    """
    ノンリニアRGBからリニアRGBへの変換
    rgbは(0,1)にして渡す
    """
    linear_image = np.where(image <= 0.04045, 
                            image / 12.92, 
                            ((image + 0.055) / 1.055) ** 2.4)
    return linear_image

def linear_rgb_to_non_linear_rgb(image):
    """
    リニアRGBからノンリニアRGBへの変換
    rgbは(0,1)にして渡す
    """
    non_linear_image = np.where(image <= 0.0031308, 
                          image * 12.92, 
                          1.055 * np.power(image, 1/2.4) - 0.055)
    return non_linear_image

def rgb_to_lab_pixel(rgb):
    """
    RGBからL*a*b*色空間への変換を行う。
    rgbは[0, 255]の範囲の値を持つ3要素のリストまたはNumPy配列。
    """
    # RGBを[0, 1]の範囲に正規化
    rgb = rgb / 255.0
    
    # sRGBからリニアRGBへの変換
    def gamma_correction(channel):
        return np.where(channel > 0.04045, ((channel + 0.055) / 1.055) ** 2.4, channel / 12.92)
    
    rgb_linear = gamma_correction(rgb)
    
    # リニアRGBからXYZへの変換
    mat_rgb_to_xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = np.dot(mat_rgb_to_xyz, rgb_linear)
    
    # XYZからL*a*b*への変換
    def xyz_to_lab(t):
        delta = 6/29
        return np.where(t > delta ** 3, t ** (1/3), t / (3 * delta ** 2) + 4/29)
    
    xyz_ref_white = np.array([0.95047, 1.00000, 1.08883])
    xyz_normalized = xyz / xyz_ref_white
    
    L = 116 * xyz_to_lab(xyz_normalized[1]) - 16
    a = 500 * (xyz_to_lab(xyz_normalized[0]) - xyz_to_lab(xyz_normalized[1]))
    b = 200 * (xyz_to_lab(xyz_normalized[1]) - xyz_to_lab(xyz_normalized[2]))
    
    return np.array([L, a, b])

def rgb_to_lab_linear(image):
    """
    RGBからL*a*b*色空間への変換を行う。
    imageは[0, 255]の範囲の値を持ち、linearにしてから処理を行う
    """
    # RGBを[0, 1]の範囲に正規化
    rgb = image / 255.0
    
    # ノンリニアRGBからリニアRGBへの変換
    def gamma_correction(channel):
        return np.where(channel > 0.04045, ((channel + 0.055) / 1.055) ** 2.4, channel / 12.92)
    
    rgb_linear = gamma_correction(rgb)
    
    # リニアRGBからXYZへの変換
    mat_rgb_to_xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = np.dot(rgb_linear, mat_rgb_to_xyz.T)
    
    # XYZからL*a*b*への変換
    def xyz_to_lab(t):
        delta = 6/29
        return np.where(t > delta ** 3, t ** (1/3), t / (3 * delta ** 2) + 4/29)
    
    xyz_ref_white = np.array([0.95047, 1.00000, 1.08883])
    xyz_normalized = xyz / xyz_ref_white
    
    L = 116 * xyz_to_lab(xyz_normalized[..., 1]) - 16
    a = 500 * (xyz_to_lab(xyz_normalized[..., 0]) - xyz_to_lab(xyz_normalized[..., 1]))
    b = 200 * (xyz_to_lab(xyz_normalized[..., 1]) - xyz_to_lab(xyz_normalized[..., 2]))
    
    return np.stack([L, a, b], axis=-1)

def lab_to_rgb_linear(lab):
    """
    L*a*b*からRGB色空間への逆変換を行う。
    labはL, a, bの値を持つ3要素のリストまたはNumPy配列。
    """
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    # L*a*b*からXYZへの変換
    def lab_to_xyz(l, a, b):
        y = (l + 16) / 116
        x = a / 500 + y
        z = y - b / 200

        xyz = np.stack((x, y, z), axis=-1)
        mask = xyz > 6/29
        xyz[mask] = xyz[mask] ** 3
        xyz[~mask] = (xyz[~mask] - 16/116) / 7.787

        # D65光源の参照白
        xyz_ref_white = np.array([0.95047, 1.00000, 1.08883])
        xyz = xyz * xyz_ref_white
        return xyz

    # 各ピクセルに適用
    xyz = np.array([lab_to_xyz(L[i, j], a[i, j], b[i, j]) for i in range(L.shape[0]) for j in range(L.shape[1])])
    xyz = xyz.reshape(L.shape[0], L.shape[1], 3)

    # XYZからリニアRGBへの変換
    mat_xyz_to_rgb = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])

    # 変換を各ピクセルに適用
    rgb_linear = np.einsum('ij, ...j', mat_xyz_to_rgb, xyz)

    # 閾値を設定
    epsilon = 1e-5

    # 極めて小さい負の値を0に設定し、その他の負の値については警告を出す
    small_neg_mask = (rgb_linear < 0) & (rgb_linear >= -epsilon)
    if np.any(rgb_linear < -epsilon):
        warnings.warn("Negative values encountered in RGB conversion")
    rgb_linear[small_neg_mask] = 0

    # リニアRGBからノンリニアRGBへのガンマ補正
    def gamma_correction(channel):
        return np.where(channel > 0.0031308, 1.055 * (channel ** (1/2.4)) - 0.055, 12.92 * channel)

    rgb = gamma_correction(rgb_linear)

    # RGB値を[0, 255]の範囲にクリッピングして整数に変換
    rgb_clipped = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    return rgb_clipped

def rgb_to_lab_non_linear(image):
    """
    RGBからL*a*b*色空間への変換を行う。
    imageは[0, 255]の範囲の値を持ち、non_linearとして扱う
    """
    # RGBを[0, 1]の範囲に正規化
    rgb = image / 255.0
    # print(rgb)
    
    # ノンリニアRGBからXYZへの変換
    mat_rgb_to_xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = np.dot(rgb, mat_rgb_to_xyz.T)
    
    # XYZからL*a*b*への変換
    def xyz_to_lab(t):
        delta = 6/29
        return np.where(t > delta ** 3, t ** (1/3), t / (3 * delta ** 2) + 4/29)
    
    xyz_ref_white = np.array([0.95047, 1.00000, 1.08883])
    xyz_normalized = xyz / xyz_ref_white
    
    L = 116 * xyz_to_lab(xyz_normalized[..., 1]) - 16
    a = 500 * (xyz_to_lab(xyz_normalized[..., 0]) - xyz_to_lab(xyz_normalized[..., 1]))
    b = 200 * (xyz_to_lab(xyz_normalized[..., 1]) - xyz_to_lab(xyz_normalized[..., 2]))
    
    return np.stack([L, a, b], axis=-1)

def lab_to_rgb_non_linear(lab):
    """
    L*a*b*からRGB色空間への逆変換を行う。
    labはL, a, bの値を持つ3要素のリストまたはNumPy配列。
    rgbはfloat32で返却
    """
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    # L*a*b*からXYZへの変換
    def lab_to_xyz(l, a, b):
        y = (l + 16) / 116
        x = a / 500 + y
        z = y - b / 200

        xyz = np.stack((x, y, z), axis=-1)
        mask = xyz > 6/29
        xyz[mask] = xyz[mask] ** 3
        xyz[~mask] = (xyz[~mask] - 16/116) / 7.787

        # D65光源の参照白
        xyz_ref_white = np.array([0.95047, 1.00000, 1.08883])
        xyz = xyz * xyz_ref_white
        return xyz

    xyz = lab_to_xyz(L, a, b)

    # XYZからノンリニアRGBへの変換
    mat_xyz_to_rgb = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])

    # 変換を各ピクセルに適用
    rgb_non_linear = np.einsum('ij, ...j', mat_xyz_to_rgb, xyz)

    # 閾値を設定
    epsilon = 1e-3

    # 極めて小さい負の値を0に設定し、その他の負の値については警告を出す
    small_neg_mask = (rgb_non_linear < 0) & (rgb_non_linear >= -epsilon)
    if np.any(rgb_non_linear < -epsilon):
        warnings.warn("Negative values encountered in RGB conversion")
    rgb_non_linear[small_neg_mask] = 0

    # RGB値を[0, 255]の範囲にクリッピングして整数に変換
    rgb_clipped = np.clip(rgb_non_linear * 255, 0, 255).astype(np.float32)

    return rgb_clipped

def lab_to_lch(lab_image):
    # Labチャンネルを分割
    L, a, b = cv2.split(lab_image)
    
    # Lはそのまま
    L = L.astype(np.float32)
    
    # C（彩度）を計算
    C = np.sqrt(a**2 + b**2)
    
    # H（色相）を計算
    H = np.arctan2(b, a)
    H = np.degrees(H)
    H = np.where(H < 0, H + 360, H)
    
    return cv2.merge([L, C, H])

def rgb_to_hsl_non_linear(image):
    # imageは(高さ, 幅, 3)の形状のNumPy配列と仮定
    # dtypeをfloatに変換して計算を行う
    image = image.astype(np.float32) / 255.0

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    max_color = np.maximum(np.maximum(R, G), B)
    min_color = np.minimum(np.minimum(R, G), B)

    L = (max_color + min_color) / 2

    delta = max_color - min_color
    S = np.zeros_like(L)
    
    # 彩度の計算
    S[delta != 0] = delta[delta != 0] / (1 - np.abs(2 * L[delta != 0] - 1))

    H = np.zeros_like(L)
    # 色相の計算
    # Rが最大値
    idx = (max_color == R) & (delta != 0)
    H[idx] = 60 * (((G[idx] - B[idx]) / delta[idx]) % 6)

    # Gが最大値
    idx = (max_color == G) & (delta != 0)
    H[idx] = 60 * (((B[idx] - R[idx]) / delta[idx]) + 2)

    # Bが最大値
    idx = (max_color == B) & (delta != 0)
    H[idx] = 60 * (((R[idx] - G[idx]) / delta[idx]) + 4)

    # 彩度と輝度をパーセンテージに変換
    S = S * 100
    L = L * 100

    return np.stack([H, S, L], axis=-1)

def rgb_to_hsl_linear(image):
    # imageは(高さ, 幅, 3)の形状のNumPy配列と仮定
    # dtypeをfloatに変換して計算を行う
    """
    linearにしてから処理を行う。
    引数は(0,255)
    返り値は0~360,0~100,0~100
    """
    rgb = image.astype(np.float32) / 255.0

    # ノンリニアRGBからリニアRGBへの変換
    def gamma_correction(channel):
        return np.where(channel > 0.04045, ((channel + 0.055) / 1.055) ** 2.4, channel / 12.92)
    
    rgb_linear = gamma_correction(rgb)

    R = rgb_linear[:, :, 0]
    G = rgb_linear[:, :, 1]
    B = rgb_linear[:, :, 2]

    max_color = np.maximum(np.maximum(R, G), B)
    min_color = np.minimum(np.minimum(R, G), B)

    L = (max_color + min_color) / 2

    delta = max_color - min_color
    S = np.zeros_like(L)
    
    # 彩度の計算
    S[delta != 0] = delta[delta != 0] / (1 - np.abs(2 * L[delta != 0] - 1))

    H = np.zeros_like(L)
    # 色相の計算
    # Rが最大値
    idx = (max_color == R) & (delta != 0)
    H[idx] = 60 * (((G[idx] - B[idx]) / delta[idx]) % 6)

    # Gが最大値
    idx = (max_color == G) & (delta != 0)
    H[idx] = 60 * (((B[idx] - R[idx]) / delta[idx]) + 2)

    # Bが最大値
    idx = (max_color == B) & (delta != 0)
    H[idx] = 60 * (((R[idx] - G[idx]) / delta[idx]) + 4)

    # 彩度と輝度をパーセンテージに変換
    S = S * 100
    L = L * 100

    return np.stack([H, S, L], axis=-1)

def rgb_to_hsl_pixel(r, g, b):
    """
    RGBをHSLに変換
    """
    R = r / 255.0
    G = g / 255.0
    B = b / 255.0

    max_color = max(R, G, B)
    min_color = min(R, G, B)
    L = (max_color + min_color) / 2.0

    if max_color == min_color:
        S = 0
        H = 0
    else:
        delta = max_color - min_color
        S = delta / (1 - abs(2 * L - 1))
        if max_color == R:
            H = 60 * (((G - B) / delta) % 6)
        elif max_color == G:
            H = 60 * (((B - R) / delta) + 2)
        elif max_color == B:
            H = 60 * (((R - G) / delta) + 4)

    return (H, S * 100, L * 100)

def hsl_to_rgb_non_linear(hsl_image):
    """
    H:degree
    S,L:0~100
    返り値は(0,255)
    """
    H, S, L = hsl_image[:, :, 0], hsl_image[:, :, 1], hsl_image[:, :, 2]
    H /= 360  # Hを0から1の範囲に正規化
    S /= 100  # Sを0から1の範囲に正規化
    L /= 100  # Lを0から1の範囲に正規化

    def hue_to_rgb(p, q, t):
        # tが0より小さい場合、1を加算
        t[t < 0] += 1
        # tが1より大きい場合、1を減算
        t[t > 1] -= 1
        # t < 1/6の場合
        r = np.copy(p)
        r[t < 1/6] = p[t < 1/6] + (q[t < 1/6] - p[t < 1/6]) * 6 * t[t < 1/6]
        # 1/6 <= t < 1/2の場合
        r[(t >= 1/6) & (t < 1/2)] = q[(t >= 1/6) & (t < 1/2)]
        # 1/2 <= t < 2/3の場合
        r[(t >= 1/2) & (t < 2/3)] = p[(t >= 1/2) & (t < 2/3)] + (q[(t >= 1/2) & (t < 2/3)] - p[(t >= 1/2) & (t < 2/3)]) * (2/3 - t[(t >= 1/2) & (t < 2/3)]) * 6
        # t >= 2/3の場合、rは変更なし（pの値を保持）
        
        return r

    rgb_image = np.zeros_like(hsl_image)
    q = np.where(L < 0.5, L * (1 + S), L + S - L * S)
    p = 2 * L - q

    rgb_image[:, :, 0] = hue_to_rgb(p, q, H + 1/3)  # R
    rgb_image[:, :, 1] = hue_to_rgb(p, q, H)        # G
    rgb_image[:, :, 2] = hue_to_rgb(p, q, H - 1/3)  # B

    return np.clip(rgb_image * 255, 0, 255).astype(np.float32)

def hsl_to_rgb_linear(hsl_image):
    """
    H:degree
    S,L:0~100
    返り値は(0,255)
    最後にノンリニア、(0,255)にして返す。(リニアが来ているとして処理)
    """
    H, S, L = hsl_image[:, :, 0], hsl_image[:, :, 1], hsl_image[:, :, 2]
    H /= 360  # Hを0から1の範囲に正規化
    S /= 100  # Sを0から1の範囲に正規化
    L /= 100  # Lを0から1の範囲に正規化

    def hue_to_rgb(p, q, t):
        # tが0より小さい場合、1を加算
        t[t < 0] += 1
        # tが1より大きい場合、1を減算
        t[t > 1] -= 1
        # t < 1/6の場合
        r = np.copy(p)
        r[t < 1/6] = p[t < 1/6] + (q[t < 1/6] - p[t < 1/6]) * 6 * t[t < 1/6]
        # 1/6 <= t < 1/2の場合
        r[(t >= 1/6) & (t < 1/2)] = q[(t >= 1/6) & (t < 1/2)]
        # 1/2 <= t < 2/3の場合
        r[(t >= 1/2) & (t < 2/3)] = p[(t >= 1/2) & (t < 2/3)] + (q[(t >= 1/2) & (t < 2/3)] - p[(t >= 1/2) & (t < 2/3)]) * (2/3 - t[(t >= 1/2) & (t < 2/3)]) * 6
        # t >= 2/3の場合、rは変更なし（pの値を保持）
        
        return r

    rgb_image = np.zeros_like(hsl_image)
    q = np.where(L < 0.5, L * (1 + S), L + S - L * S)
    p = 2 * L - q

    rgb_image[:, :, 0] = hue_to_rgb(p, q, H + 1/3)  # R
    rgb_image[:, :, 1] = hue_to_rgb(p, q, H)        # G
    rgb_image[:, :, 2] = hue_to_rgb(p, q, H - 1/3)  # B

        # 閾値を設定
    epsilon = 1e-5

    # 極めて小さい負の値を0に設定し、その他の負の値については警告を出す
    small_neg_mask = (rgb_image < 0) & (rgb_image >= -epsilon)
    if np.any(rgb_image < -epsilon):
        warnings.warn("Negative values encountered in RGB conversion")
    rgb_image[small_neg_mask] = 0

    # ガンマ補正
    def gamma_correction(channel):
        return np.where(channel > 0.0031308, 1.055 * (channel ** (1/2.4)) - 0.055, 12.92 * channel)

    rgb = gamma_correction(rgb_image)

    # RGB値を[0, 255]の範囲にクリッピングして整数に変換
    rgb_clipped = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    return rgb_clipped

def hsl_to_mhsl(hsl_image):
    H, S, L = hsl_image[:, :, 0], (hsl_image[:, :, 1] / 100), (hsl_image[:, :, 2] / 100)
    R, E, G, C, B, M = 0.30, 0.66, 0.59, 0.64, 0.12, 0.26


    # Hの値に基づいてqとtを計算
    q = (H / 60).astype(int)
    t = H % 60

    a = [R, E, G, C, B, M, R]

    # alpha, l_fun_smax, l_funの計算
    alpha = np.take(a, q + 1) * (t / 60.0) + np.take(a, q) * (60.0 - t) / 60.0
    l_fun_smax = -np.log2(alpha)
    l_fun = l_fun_smax * S + (1.0 - S)

    # l_tildaの計算とh_tilda, s_tildaの設定
    l_tilda = 100 * (L ** l_fun)
    h_tilda = H
    s_tilda = S * 100

    # 修正されたHSL値を含む新しい画像を返す
    mhsl_image = np.stack((h_tilda, s_tilda, l_tilda), axis=-1)
    return mhsl_image

def hsl_to_mhsl_pixel(H, S, L):
    """
    HSLから修正されたHSL（mHSL）に変換する
    """
    R, E, G, C, B, M = 0.30, 0.66, 0.59, 0.64, 0.12, 0.26

    # Hの値に基づいてqとtを計算
    q = int(H / 60)
    t = H % 60

    a = [R, E, G, C, B, M, R]

    # alpha, l_fun_smax, l_funの計算
    alpha = a[q + 1] * (t / 60.0) + a[q] * (60.0 - t) / 60.0
    l_fun_smax = -np.log2(alpha)
    l_fun = l_fun_smax * (S / 100.0) + (1.0 - (S / 100.0))

    # l_tildaの計算とh_tilda, s_tildaの設定
    l_tilda = 100 * (L / 100.0) ** l_fun
    h_tilda = H
    s_tilda = S

    # 修正されたHSL値を返す
    return h_tilda, s_tilda, l_tilda

def mhsl_to_hsl(mhsl_image):
    # 配列からHSLの各成分を取得
    H_tilda, S_tilda, L_tilda = mhsl_image[:,:,0], (mhsl_image[:,:,1] / 100), (mhsl_image[:,:,2] / 100)

    R, E, G, C, B, M = 0.30, 0.66, 0.59, 0.64, 0.12, 0.26
    # R, E, G, C, B, M = 0.30, 0.66, 0.59, 0.64, 0.40, 0.60
    a = np.array([R, E, G, C, B, M, R])

    q = (H_tilda / 60).astype(int)
    t = H_tilda % 60

    alpha = a[q + 1] * (t / 60.0) + a[q] * (60.0 - t) / 60.0
    l_fun_smax = -np.log2(alpha)
    l_fun = l_fun_smax * S_tilda + (1.0 - S_tilda)

    L_org = np.power(L_tilda, (1.0 / l_fun)) * 100

    # HとSは変換されていないため、そのまま返す
    H_org, S_org = H_tilda, (S_tilda * 100)

    # 変換後の画像データを構築
    hsl_image = np.stack((H_org, S_org, L_org), axis=-1)
    return hsl_image

def hsl_to_cartesian(image):
    """
    HSL色空間で表された画像を直交座標系に変換する。
    :param image: HSL色空間の画像 (高さ x 幅 x 3のNumPy配列)
    :return: 直交座標系に変換された画像 (高さ x 幅 x 3のNumPy配列)
    """
    height, width, _ = image.shape
    cartesian_image = np.zeros_like(image, dtype=float)
    
    h_rad = np.deg2rad(image[:, :, 0])  # 色相をラジアンに変換
    s = image[:, :, 1]  # 彩度
    l = image[:, :, 2]  # 輝度
    
    cartesian_image[:, :, 0] = s * np.cos(h_rad)  # x
    cartesian_image[:, :, 1] = s * np.sin(h_rad)  # y
    cartesian_image[:, :, 2] = l  # z
    
    return cartesian_image

def cartesian_to_hsl(cartesian_image):
    """
    直交座標系に変換されたHSL色空間の画像を通常のHSLに戻す。
    :param cartesian_image: 直交座標系に変換された画像 (高さ x 幅 x 3のNumPy配列)
    :return: HSL色空間の画像 (高さ x 幅 x 3のNumPy配列)
    """
    height, width, _ = cartesian_image.shape
    hsl_image = np.zeros_like(cartesian_image, dtype=float)
    
    # xとy座標から色相(H)と彩度(S)を計算
    x = cartesian_image[:, :, 0]
    y = cartesian_image[:, :, 1]
    hsl_image[:, :, 0] = (np.arctan2(y, x) * (180 / np.pi)) % 360  # 色相H
    hsl_image[:, :, 1] = np.sqrt(x**2 + y**2)  # 彩度S
    
    # z座標は輝度(L)に直接対応
    hsl_image[:, :, 2] = cartesian_image[:, :, 2]  # 輝度L
    
    return hsl_image