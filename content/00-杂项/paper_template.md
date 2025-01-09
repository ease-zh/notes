<!-- 标题 -->

<!-- Meta Data -->
<table>

  <!-- 作者 -->

  <tr>

    <td style="color:#193c47; background-color:#dbeedd; padding:8px;">

      <b>作者:</b> ${topItem.getCreators().slice(0, 10).map((v) => v.firstName + " " + v.lastName).join("; ") + (topItem.getCreators().length > 10 ? "; et al." : ";")}

    </td>

  </tr>

  

  <!-- 期刊 -->

  <tr>

    <td style="color:#193c47; background-color:#f3faf4; padding:8px;">

      <b style="color:#193c47;">期刊: <b style="color:#FF0000">${topItem.getField('publicationTitle')}</b></b><b style="color:#193c47;"> （发表日期: ${topItem.getField("date").split('T')[0]}）</b>

    </td>

  </tr>

  

  <!-- 期刊分区 -->

  <tr>

    <td style="color:#193c47; background-color:#dbeedd; padding:8px;">

      <b>期刊分区: </b>

      <!-- Zotero7中，引用了Ethereal Style插件的标签，请提前安装Ethereal Style-->

      ${{

      let space = " ㅤㅤ ㅤㅤ"

      return Array.prototype.map.call(

        Zotero.ZoteroStyle.api.renderCell(topItem, "publicationTags").childNodes,

        e => {

          e.innerText =  space + e.innerText + space;

          return e.outerHTML

        }

        ).join(space)

      }}$

    </td>

  </tr>

  

  <!-- 本地链接 -->

  <tr>

    <td style="color:#193c47; background-color:#f3faf4; padding:8px;">

      ${(() => {

        const attachments = Zotero.Items.get(topItem.getAttachments());

        const pdf = attachments.filter((i) => i.isPDFAttachment());

        if (pdf && pdf.length > 0) {

          return `<b>本地链接: </b><a href="zotero://open-pdf/0_${pdf[0].key}">${pdf[0].getFilename()}</a>`;

        } else if (attachments && attachments.length > 0) {

          return `<b>本地链接: </b><a href="zotero://open-pdf/0_${attachments[0].key}">${attachments[0].getFilename()}</a>`;

        } else {

          return `<b>本地链接: </b>`;

        }

      })()}

    </td>

  </tr>

  

  <!-- DOI or URL -->

  <tr>

    <td style="color:#193c47; background-color:#dbeedd; padding:8px;">

      ${(() => {

        const doi = topItem.getField("DOI");

        if (doi) {

          return `<b>DOI: </b><a href="https://doi.org/${topItem.getField('DOI')}">${topItem.getField('DOI')}</a>`;

        } else {

          return `<b>URL: </b><a href="${topItem.getField('url')}">${topItem.getField('url')}</a>`;

        }

      })()}

    </td>

  </tr>

  <!-- 摘要 -->

  <tr>

    <td style="color:#193c47; background-color:#f3faf4; padding:8px;">

      ${(() => {

        const abstractTranslation = topItem.getField('abstractTranslation');

        if (abstractTranslation) {

          return `<b>摘要翻译: </b><i>${abstractTranslation}</i>`;

        } else {

          return `<b>摘要: </b><i>${topItem.getField('abstractNote')}</i>`;

        }

      })()}

    </td>

  </tr>

  

  <!-- 笔记日期 -->

  <tr>

    <td style="color:#193c47; background-color:#dbeedd; padding:8px;">

      <b>笔记日期: </b>${new Date().toLocaleString()}

    </td>

  </tr>

  

</table>

  

<!-- 正文 -->

<span>

  <h2 style="color:#e0ffff; background-color:#66cdaa;">📜 研究核心</h2>

  <hr />

</span>

<blockquote>Tips: 做了什么，解决了什么问题，创新点与不足？</blockquote>

<p></p>

<h3>⚙️ 内容</h3>

<p></p>

<h3>💡 创新点</h3>

<p></p>

<h3>🧩 不足</h3>

<p></p>

  

<span>

  <h2 style="color:#20b2aa; background-color:#afeeee;">🔁 研究内容</h2>

  <hr />

</span>

<p></p>

<h3>💧 数据</h3>

<p></p>

<h3>👩🏻‍💻 方法</h3>

<p></p>

<h3>🔬 实验</h3>

<p></p>

<h3>📜 结论</h3>

<p></p>

  

<span>

  <h2 style="color:#004d99; background-color:#87cefa;">🤔 个人总结</h2>

  <hr />

</span>

<blockquote>Tips: 你对哪些内容产生了疑问，你认为可以如何改进？</blockquote>

<p></p>

<h3>🙋‍♀️ 重点记录</h3>

<p></p>

<h3>📌 待解决</h3>

<p></p>

<h3>💭 思考启发</h3>

<p></p>