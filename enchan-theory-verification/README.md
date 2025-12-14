# Enchan Theory Verification Package (v0.3.0)

このパッケージは、**Enchan Theory v0.3.0 (Inception/Defect Model)** において理論的に導出された加速度スケーリング則（Quadrature Closure）が、実際の銀河観測データ（SPARC）と整合することを検証するためのものです。

## 理論的背景 (v0.3.0 Pivot)

Enchan Theory v0.3.0 では、時空を「トポロジカル欠陥（時間遅延場 S の皺）」を含む構造としてモデル化しました。
そのラグランジアンから導かれる理論予測は以下の通りです：

1.  **幾何学的暗黒物質:** 欠陥のエネルギー密度は rho ~ 1/r^2 となり、平坦な回転曲線を生む。
2.  **加速度スケール a0:** 真空の剛性 sigma_vac と欠陥の強さによって決まる派生パラメータである。
3.  **有効加速度則:** 重力加速度 g_tot は、バリオン重力 g_bar と欠陥の相互作用により、以下の「幾何学的閉包」に従う。

    g_tot = sqrt( g_bar^2 + a0 * g_bar )

このパッケージに含まれるスクリプトは、上記の **「3. 有効加速度則」** が、現実の宇宙において高い精度で成立していることを定量的に証明します。

## 含まれるファイル

* `enchan_core_model.py`: 理論モデル（幾何学的閉包）の定義コア
* `enchan_btfr_reproduce_enchan.py`: BTFRデータから a0 の普遍性を検証
* `enchan_rar_reproduce_enchan.py`: RAR（放射加速度関係）における理論曲線の適合度を検証
* `enchan_rotationcurve_reproduce_enchan.py`: 個々の銀河の回転曲線を予測・検証

## 入力データ（別途用意）

* `Rotmod_LTG.zip` (SPARC mass models)
* `BTFR_Lelli2019.mrt` (SPARC BTFR table)

## 実行方法

1. **a0 の特定:**
    BTFRデータから、理論パラメータ a0 の観測値を逆算します。
    ```bash
    python enchan_btfr_reproduce_enchan.py --mrt BTFR_Lelli2019.mrt
    ```
(出力例: median a0 = 1.2e-10 m/s^2)

2. **理論の検証:**
    特定した a0 を用いて、RARおよび回転曲線の再現性をテストします。
    ```bash
    python enchan_rar_reproduce_enchan.py --zip Rotmod_LTG.zip --a0 1.2e-10
    python enchan_rotationcurve_reproduce_enchan.py --zip Rotmod_LTG.zip --a0 1.2e-10
    ```