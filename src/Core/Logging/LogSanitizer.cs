using System;
using System.CodeDom.Compiler;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using System.Text.RegularExpressions.Generated;

namespace MegaCrit.Sts2.Core.Logging;

public static partial class LogSanitizer
{
	private static readonly string _homeReplacement = (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "%USERPROFILE%" : "~");

	[GeneratedRegex("\\b76561\\d{12}\\b")]
	private static partial Regex SteamIdRegex();

	public static string Sanitize(string text)
	{
		string folderPath = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
		if (!string.IsNullOrEmpty(folderPath))
		{
			text = text.Replace(folderPath, _homeReplacement);
			string text2 = folderPath.Replace('\\', '/');
			if (text2 != folderPath)
			{
				text = text.Replace(text2, _homeReplacement);
			}
		}
		text = SteamIdRegex().Replace(text, ReplaceSteamId);
		return text;
	}

	public static string ReplaceSteamId(Match m)
	{
		ulong id = ulong.Parse(m.Value);
		string text = IdAnonymizer.Anonymize(id).ToString();
		return "A" + text;
	}
}
