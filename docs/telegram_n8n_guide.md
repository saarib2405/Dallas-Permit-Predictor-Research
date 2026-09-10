# Connecting Dallas Permit Predictor to Telegram via n8n

This guide will walk you through setting up a Telegram bot, integrating it into an n8n AI Agent via a secure ngrok tunnel, and connecting it to your deployed Dallas Permit Predictor API.

## Step 1: Create a Telegram Bot

You need a bot token from Telegram to allow n8n to receive and send messages on your behalf.

1. Open **Telegram** on your phone or desktop.
2. Search for **@BotFather** (with the blue verified checkmark) and start a chat.
3. Send the message `/newbot`.
4. BotFather will ask for a **Name** (e.g., *Dallas Permit Predictor*).
5. BotFather will then ask for a **Username** (must end in `bot`, e.g., *DallasPermit_bot*).
6. BotFather will reply with a congratulatory message containing your **HTTP API Token**.
   > [!IMPORTANT]
   > Copy this Token and keep it secret!

## Step 2: Set up ngrok Secure Tunnel

Because Telegram strictly requires an HTTPS server to send its webhooks, and your n8n is running locally, we use **ngrok** to provide an enterprise-grade secure tunnel to your local machine.

1. Go to **[ngrok.com/download](https://ngrok.com/download)** and download the Windows zip.
2. Extract the zip file to get `ngrok.exe`.
3. Double-click `ngrok.exe` to open the terminal.
4. Type `ngrok http 5678` and hit Enter.
5. In the output, look for the **Forwarding** line (it will look like `https://1a2b-3c4d.ngrok-free.app`). 
   > [!IMPORTANT]
   > Copy that `https://...` address. Leave this black terminal window open in the background!

## Step 3: Start n8n with Webhooks Enabled

Now you must tell n8n to build its webhooks using that secure ngrok URL.

1. Open your normal Windows Command Prompt (`cmd`).
2. Type `set WEBHOOK_URL=https://[YOUR-NGROK-URL-HERE]` (replace with your copied URL) and press Enter.
3. In that precise same command prompt window, type `npx n8n start` and press Enter.
4. Open your browser and go to `http://localhost:5678`.

## Step 4: Import the Workflow & Credentials

1. In n8n, go to **Credentials** on the left menu. Click **Add Credential**.
2. Add your **Telegram API** token (from Step 1) and your **Google Gemini API** (or OpenAI) key.
3. Go back to the main canvas and click **Add Workflow**.
4. Click the `...` menu in the top right and select **Import from File**.
5. Select the `dallas_permit_agent_workflow.json` from your project folder.
6. Double-click the **Telegram Trigger** node and the **Telegram Reply** node (at the end) and attach your Telegram credentials.
7. Double-click the **Language Model** node and attach your Gemini/OpenAI credentials.
8. **Save** the workflow.

## Step 5: Test the Bot!

1. Click the big **Test Workflow** button at the bottom of the screen.
2. Open Telegram and send a message to your bot: *"How much would a 15,000 sqft commercial office building cost to construct in 75201?"*
3. n8n will catch the incoming webhook via ngrok, the AI Agent will extract the data, hit the Render API securely, and format a human-readable reply.
4. Your bot will send the response back in Telegram naturally!
