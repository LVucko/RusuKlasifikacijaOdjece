package com.example.clothingclassificator
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import android.hardware.Camera
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.Button
import android.widget.TextView
import java.io.IOException
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity(), SurfaceHolder.Callback, Camera.PictureCallback {
    private lateinit var camera: Camera
    private lateinit var surfaceView: SurfaceView
    private lateinit var surfaceHolder: SurfaceHolder
    private lateinit var captureButton: Button
    private lateinit var predictionTextView: TextView
    private lateinit var tfliteInterpreter: Interpreter
    private val CAMERA_PERMISSION_REQUEST_CODE = 100

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val tfliteModel = loadModelFromAsset() // Load the TFLite model from asset folder
        val options = Interpreter.Options()
        tfliteInterpreter = Interpreter(tfliteModel, options)
        surfaceView = findViewById(R.id.cameraSurfaceView)
        captureButton = findViewById(R.id.captureButton)
        predictionTextView = findViewById(R.id.predictionTextView)

        surfaceHolder = surfaceView.holder
        surfaceHolder.addCallback(this)

        captureButton.setOnClickListener {
            camera.takePicture(null, null, this)
        }
    }
    private fun loadModelFromAsset(): ByteBuffer {
        val assetManager = assets
        val modelDescriptor = assetManager.openFd("ClothingModel.tflite")
        val inputStream = FileInputStream(modelDescriptor.fileDescriptor)
        val modelData = ByteBuffer.allocateDirect(modelDescriptor.declaredLength.toInt())
        inputStream.channel.use { channel ->
            modelData.put(channel.map(FileChannel.MapMode.READ_ONLY, modelDescriptor.startOffset, modelDescriptor.declaredLength))
        }
        modelData.order(ByteOrder.nativeOrder())
        return modelData
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_REQUEST_CODE
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera()
            }
        }
    }

    private fun startCamera() {
        try {
            camera = Camera.open()
            camera.setPreviewDisplay(surfaceHolder)
            camera.startPreview()
            camera.setDisplayOrientation(90)
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    override fun surfaceCreated(holder: SurfaceHolder) {}

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        if (holder.surface == null) {
            return
        }
        try {
            camera.stopPreview()
            camera.setPreviewDisplay(holder)
            camera.startPreview()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        camera.stopPreview()
        camera.release()
    }

    override fun onPictureTaken(data: ByteArray?, camera: Camera?) {
        val bitmap = BitmapFactory.decodeByteArray(data, 0, data?.size ?: 0)
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 100, 100, false)
        val normalizedBitmap = resizedBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val modelInputBuffer = convertBitmapToByteBuffer(normalizedBitmap)

        val outputBuffer = Array(1) { FloatArray(NUM_CLASSES) }
        tfliteInterpreter.run(modelInputBuffer, outputBuffer)

        val predictedClass = getPredictedClass(outputBuffer[0])
        predictionTextView.text = predictedClass

        camera?.startPreview()
    }
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val inputShape = tfliteInterpreter.getInputTensor(0).shape() // Get the input tensor shape
        val inputSize = inputShape[1] * inputShape[2] * inputShape[3]
        val modelInputBuffer = ByteBuffer.allocateDirect(inputSize * 4).apply {
            order(ByteOrder.nativeOrder())
            rewind()
        }

        val intValues = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (pixelValue in intValues) {
            val r = (pixelValue shr 16) and 0xFF
            val g = (pixelValue shr 8) and 0xFF
            val b = pixelValue and 0xFF

            val normalizedR = r / 255.0f
            val normalizedG = g / 255.0f
            val normalizedB = b / 255.0f

            modelInputBuffer.putFloat(normalizedR)
            modelInputBuffer.putFloat(normalizedG)
            modelInputBuffer.putFloat(normalizedB)
        }

        return modelInputBuffer
    }
    val NUM_CLASSES = 15
    val classLabels = arrayOf("Blazer", "Blouse", "Body", "Hat", "Hoodie", "Longsleeve", "Outwear", "Pants", "Polo", "Shirt", "Shoes", "Skirt", "Top", "Tshirt", "Undershirt")

    private fun getPredictedClass(output: FloatArray): String {
        val predictedClassIndex = output.indices.maxByOrNull { output[it] } ?: 0
        return classLabels[predictedClassIndex]
    }
}
