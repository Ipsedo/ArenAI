package com.samuelberrien.arenai.new_set_controls;

import android.content.Context;
import android.content.SharedPreferences;
import android.view.InputDevice;
import android.view.MotionEvent;
import android.view.View;

import androidx.annotation.Nullable;

public class ContinuousInputListener implements SharedPreferences.OnSharedPreferenceChangeListener, View.OnGenericMotionListener {

    public final static String CONTINUOUS_INPUTS_SHARED_PREF = "ContinuousInputsSharedPref";

    private int inputId;
    private final String inputSharedPreferenceName;

    private float value;

    public ContinuousInputListener(Context context, String sharedPreferenceName) {
        inputSharedPreferenceName = sharedPreferenceName;

        SharedPreferences pref = context.getSharedPreferences(CONTINUOUS_INPUTS_SHARED_PREF, Context.MODE_PRIVATE);
        pref.registerOnSharedPreferenceChangeListener(this);
        inputId = pref.getInt(inputSharedPreferenceName, -1);
        value = 0.f;
    }

    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, @Nullable String key) {
        inputId = sharedPreferences.getInt(inputSharedPreferenceName, -1);
    }

    @Override
    public boolean onGenericMotion(View v, MotionEvent event) {
        if ((event.getSource() & InputDevice.SOURCE_GAMEPAD) ==
                InputDevice.SOURCE_GAMEPAD &&
                event.getAction() == MotionEvent.ACTION_MOVE && (
                event.getSource() & inputId
        ) == inputId) {
            value = event.getAxisValue(inputId);
            return true;
        }
        return false;
    }

    public float getValue() {
        return value;
    }
}
