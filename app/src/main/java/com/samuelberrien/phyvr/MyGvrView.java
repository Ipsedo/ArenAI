package com.samuelberrien.phyvr;

import android.content.Context;
import android.content.res.AssetManager;
import android.opengl.Matrix;
import android.util.AttributeSet;

import com.google.vr.sdk.base.Eye;
import com.google.vr.sdk.base.GvrView;
import com.google.vr.sdk.base.HeadTransform;
import com.google.vr.sdk.base.Viewport;

import javax.microedition.khronos.egl.EGLConfig;

public class MyGvrView extends GvrView implements GvrView.StereoRenderer {

    public MyGvrView(Context context) {
        super(context);
        setRenderer(this);
    }

    public MyGvrView(Context context, AttributeSet attrs) {
        super(context, attrs);
        setRenderer(this);
    }

    @Override
    public void onPause(){
        super.onPause();
    }

    @Override
    public void onResume() {
        super.onResume();
    }

    /**
     * Stereo Renderer stuff
     */

    // TODO wrappers pour rendu niveau

    private final float Z_NEAR = 1f;
    private final float Z_FAR = 50f;

    static {
        System.loadLibrary("bullet");
    }

    private long levelPtr;

    @Override
    public void onNewFrame(HeadTransform headTransform) {
        float[] mHeadView = new float[16];
        headTransform.getHeadView(mHeadView, 0);
        updateLevel(levelPtr, mHeadView);
    }

    @Override
    public void onDrawEye(Eye eye) {
        float[] eyeView = eye.getEyeView();

        float[] mProjectionMatrix = eye.getPerspective(Z_NEAR, Z_FAR);
        drawLevel(levelPtr, mProjectionMatrix, eyeView, new float[4], new float[3]);
    }

    @Override
    public void onFinishFrame(Viewport viewport) {

    }

    @Override
    public void onSurfaceChanged(int width, int height) {

    }

    @Override
    public void onSurfaceCreated(EGLConfig config) {
        levelPtr = initLevel(getContext().getAssets());
    }

    @Override
    public void onRendererShutdown() {

    }

    public native long initLevel(AssetManager assetManager);

    public native void updateLevel(long levelptr, float[] mHeadView);

    public native void drawLevel(long levelptr,
                                 float[] mEyeProjectionMatrix,
                                 float[] mEyeViewMatrix,
                                 float[] myLighPosInEyeSpace,
                                 float[] mCameraPos);
}
