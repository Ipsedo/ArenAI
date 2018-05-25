package com.samuelberrien.phyvr;

import android.content.Context;
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
    }

    public MyGvrView(Context context, AttributeSet attrs) {
        super(context, attrs);
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



    @Override
    public void onNewFrame(HeadTransform headTransform) {
        float[] mHeadView = new float[16];
        headTransform.getHeadView(mHeadView, 0);
    }

    @Override
    public void onDrawEye(Eye eye) {
        float[] eyeView = eye.getEyeView();

        float[] mProjectionMatrix = eye.getPerspective(Z_NEAR, Z_FAR);
    }

    @Override
    public void onFinishFrame(Viewport viewport) {

    }

    @Override
    public void onSurfaceChanged(int width, int height) {

    }

    @Override
    public void onSurfaceCreated(EGLConfig config) {

    }

    @Override
    public void onRendererShutdown() {

    }
}
