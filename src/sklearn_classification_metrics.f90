! Weighted classification summaries: sklearn 1.6.1 metrics/_classification.py (BSD).
! NumPy 2.2.6 pairwise reductions (BSD; retained numpy-COPYING notice).
! Native sparse accumulation, adapters and quiet IEEE composition are MIT.
module sklearn_classification_metrics
  use sklearn_common, only: dp,require
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite,ieee_value,ieee_quiet_nan
  implicit none
  private
  public :: precision_score,recall_score,f1_score,jaccard_score,fbeta_score
  public :: precision_recall_fscore_support,multilabel_confusion_matrix
  interface precision_score
    module procedure precision_score_vv
    module procedure precision_score_bb
    module procedure precision_score_ii
    module procedure precision_score_ll
    module procedure precision_score_ss
    module procedure precision_score_si
    module procedure precision_score_is
    module procedure precision_score_sl
    module procedure precision_score_ls
  end interface
  interface recall_score
    module procedure recall_score_vv
    module procedure recall_score_bb
    module procedure recall_score_ii
    module procedure recall_score_ll
    module procedure recall_score_ss
    module procedure recall_score_si
    module procedure recall_score_is
    module procedure recall_score_sl
    module procedure recall_score_ls
  end interface
  interface f1_score
    module procedure f1_score_vv
    module procedure f1_score_bb
    module procedure f1_score_ii
    module procedure f1_score_ll
    module procedure f1_score_ss
    module procedure f1_score_si
    module procedure f1_score_is
    module procedure f1_score_sl
    module procedure f1_score_ls
  end interface
  interface jaccard_score
    module procedure jaccard_score_vv
    module procedure jaccard_score_bb
    module procedure jaccard_score_ii
    module procedure jaccard_score_ll
    module procedure jaccard_score_ss
    module procedure jaccard_score_si
    module procedure jaccard_score_is
    module procedure jaccard_score_sl
    module procedure jaccard_score_ls
  end interface
  interface fbeta_score
    module procedure fbeta_score_vv
    module procedure fbeta_score_bb
    module procedure fbeta_score_ii
    module procedure fbeta_score_ll
    module procedure fbeta_score_ss
    module procedure fbeta_score_si
    module procedure fbeta_score_is
    module procedure fbeta_score_sl
    module procedure fbeta_score_ls
  end interface
  interface precision_recall_fscore_support
    module procedure precision_recall_fscore_support_vv
    module procedure precision_recall_fscore_support_bb
    module procedure precision_recall_fscore_support_ii
    module procedure precision_recall_fscore_support_ll
    module procedure precision_recall_fscore_support_ss
    module procedure precision_recall_fscore_support_si
    module procedure precision_recall_fscore_support_is
    module procedure precision_recall_fscore_support_sl
    module procedure precision_recall_fscore_support_ls
  end interface
  interface multilabel_confusion_matrix
    module procedure multilabel_confusion_matrix_vv
    module procedure multilabel_confusion_matrix_bb
    module procedure multilabel_confusion_matrix_ii
    module procedure multilabel_confusion_matrix_ll
    module procedure multilabel_confusion_matrix_ss
    module procedure multilabel_confusion_matrix_si
    module procedure multilabel_confusion_matrix_is
    module procedure multilabel_confusion_matrix_sl
    module procedure multilabel_confusion_matrix_ls
  end interface
contains
  function descending_order(scores) result(order)
    real(dp), intent(in) :: scores(:)
    integer :: order(size(scores)),work(size(scores)),n,width,left,middle,right,i,j,k
    n=size(scores);order=[(i,i=1,n)];width=1
    do while(width<n)
      left=1
      do while(left<=n)
        middle=left+min(width,n-left+1);right=middle+min(width,n-middle+1)
        i=left;j=middle
        do k=left,right-1
          if(i>=middle) then
            work(k)=order(j);j=j+1
          else if(j>=right) then
            work(k)=order(i);i=i+1
          else if(scores(order(i))>=scores(order(j))) then
            work(k)=order(i);i=i+1
          else
            work(k)=order(j);j=j+1
          end if
        end do
        left=right
      end do
      order=work
      if(width>=n-width) exit
      width=width*2
    end do
  end function
  function sorted_classes(y) result(classes)
    integer, intent(in) :: y(:)
    integer, allocatable :: classes(:)
    integer :: order(size(y)),work(size(y)),i,n
    order=descending_order(real(y,dp));n=0
    do i=size(y),1,-1
      if(n>0) then
        if(work(n)==y(order(i))) cycle
      end if
      n=n+1;work(n)=y(order(i))
    end do
    classes=work(:n)
  end function
  integer function class_position(classes,label) result(position)
    integer, intent(in) :: classes(:),label
    integer :: left,right,middle
    left=1;right=size(classes);position=0
    do while(left<=right)
      middle=left+(right-left)/2
      if(classes(middle)==label) then
        position=middle;return
      else if(classes(middle)<label) then
        left=middle+1
      else
        right=middle-1
      end if
    end do
  end function
  recursive real(dp) function numpy_sum(a) result(total)
    real(dp), intent(in) :: a(:)
    real(dp) :: partial(8),left,right
    integer :: n,i,j,whole,split
    n=size(a);total=-0._dp
    if(n<8) then
      do i=1,n
        total=ieee_difference(total,-a(i))
      end do
    else if(n<=128) then
      partial=a(:8);whole=n-mod(n,8)
      do i=9,whole,8
        do j=1,8
          partial(j)=ieee_difference(partial(j),-a(i+j-1))
        end do
      end do
      left=ieee_difference(ieee_difference(partial(1),-partial(2)),-ieee_difference(partial(3),-partial(4)))
      right=ieee_difference(ieee_difference(partial(5),-partial(6)),-ieee_difference(partial(7),-partial(8)))
      total=ieee_difference(left,-right)
      do i=whole+1,n
        total=ieee_difference(total,-a(i))
      end do
    else
      split=n/2;split=split-mod(split,8)
      total=ieee_difference(numpy_sum(a(:split)),-numpy_sum(a(split+1:)))
    end if
  end function
  real(dp) function average_values(values,weights) result(value)
    real(dp), intent(in) :: values(:)
    real(dp), optional, intent(in) :: weights(:)
    real(dp) :: products(size(values))
    integer :: i
    if(present(weights)) then
      do i=1,size(values)
        products(i)=ieee_product(values(i),weights(i))
      end do
      value=ieee_quotient(numpy_sum(products),numpy_sum(weights))
    else
      value=ieee_quotient(numpy_sum(values),real(size(values),dp))
    end if
  end function
  real(dp) function undefined() result(value)
    value=ieee_value(0._dp,ieee_quiet_nan)
  end function
  function metric_weights(n,sample_weight) result(weights)
    integer, intent(in) :: n
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp) :: weights(n)
    weights=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'classification metric sample weight length mismatch')
      weights=sample_weight
    end if
  end function
  subroutine check_zero(zero,jaccard)
    real(dp), intent(in) :: zero
    logical, intent(in) :: jaccard
    if(ieee_is_nan(zero)) then
      call require(.not.jaccard,'Jaccard zero_division must be zero or one')
    else
      call require(zero==0.or.zero==1,'zero_division must be zero, one or NaN')
    end if
  end subroutine
  subroutine check_beta(beta)
    real(dp), intent(in) :: beta
    call require(.not.ieee_is_nan(beta),'F-beta beta must be nonnegative')
    call require(beta>=0,'F-beta beta must be nonnegative')
    if(ieee_is_finite(beta)) call require(beta<=sqrt(huge(1._dp)),'F-beta beta square overflows')
  end subroutine
  function selected_labels(p,labels) result(selected)
    integer, intent(in) :: p
    integer, optional, intent(in) :: labels(:)
    integer, allocatable :: selected(:)
    integer :: j
    selected=[(j-1,j=1,p)]
    if(present(labels)) selected=labels
    call require(all(selected>=0).and.all(selected<p),'multilabel metric labels must be in [0,n_labels)')
  end function
  subroutine check_indicator(y)
    integer, intent(in) :: y(:,:)
    integer :: first,second,i,j
    logical :: seen
    call require(size(y,1)>0.and.size(y,2)>=2,'multilabel metrics require nonempty samples and at least two labels')
    first=y(1,1);second=first;seen=.false.
    do j=1,size(y,2)
      do i=1,size(y,1)
        if(y(i,j)==first) cycle
        if(.not.seen) then
          seen=.true.;second=y(i,j)
        end if
        call require(y(i,j)==second,'multilabel metrics require indicator targets with at most two values')
      end do
    end do
  end subroutine
  function indicator_csr(y) result(matrix)
    integer, intent(in) :: y(:,:)
    type(csr_matrix) :: matrix
    call check_indicator(y)
    matrix=csr_from_dense(real(merge(1,0,y/=0),dp))
  end function
  subroutine check_csr(y)
    type(csr_matrix), intent(in) :: y
    integer :: i,start,stop
    call y%validate()
    call require(y%n_rows>0.and.y%n_cols>=2,'CSR multilabel metrics require samples and at least two labels')
    call require(all(ieee_is_finite(y%data)),'CSR multilabel metric data must be finite')
    call require(all(y%data==1),'CSR multilabel metrics must store ones')
    do i=1,y%n_rows
      start=y%indptr(i);stop=y%indptr(i+1)-1
      if(stop>start) call require(all(y%indices(start+1:stop)>y%indices(start:stop-1)), &
        'CSR multilabel metric indices must be sorted and unique')
    end do
  end subroutine
  function confusion_vector(y,p,labels,w,samplewise) result(cm)
    integer, intent(in) :: y(:),p(:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: w(:)
    logical, intent(in) :: samplewise
    real(dp), allocatable :: cm(:,:,:),tp(:),predicted(:),true_count(:)
    integer, allocatable :: selected(:),classes(:)
    real(dp) :: weights(size(y)),total,fp,fn
    integer :: i,j,it,ip,index
    call require(size(y)==size(p),'classification metric target/prediction lengths differ')
    call require(.not.samplewise,'samplewise metrics require multilabel targets')
    weights=metric_weights(size(y),w);total=numpy_sum(weights)
    selected=sorted_classes([y,p]);if(present(labels)) selected=labels
    classes=sorted_classes([y,p,selected])
    allocate(tp(size(classes)),predicted(size(classes)),true_count(size(classes)));tp=0;predicted=0;true_count=0
    do i=1,size(y)
      it=class_position(classes,y(i));ip=class_position(classes,p(i))
      true_count(it)=ieee_difference(true_count(it),-weights(i));predicted(ip)=ieee_difference(predicted(ip),-weights(i))
      if(it==ip) tp(it)=ieee_difference(tp(it),-weights(i))
    end do
    allocate(cm(size(selected),2,2));cm=0
    do j=1,size(selected)
      index=class_position(classes,selected(j))
      fp=ieee_difference(predicted(index),tp(index));fn=ieee_difference(true_count(index),tp(index))
      cm(j,1,1)=ieee_difference(ieee_difference(ieee_difference(total,tp(index)),fp),fn)
      cm(j,1,2)=fp;cm(j,2,1)=fn;cm(j,2,2)=tp(index)
    end do
  end function
  function confusion_sparse(y,p,labels,w,samplewise) result(cm)
    type(csr_matrix), intent(in) :: y,p
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: w(:)
    logical, intent(in) :: samplewise
    real(dp), allocatable :: cm(:,:,:),tp(:),true_count(:),predicted(:)
    integer, allocatable :: selected(:),multiplicity(:)
    real(dp) :: weights(y%n_rows),total,fp,fn,capacity
    integer :: i,j,a,b,stop_a,stop_b,field,ntrue,npred,ntp
    call check_csr(y);call check_csr(p)
    call require(y%n_rows==p%n_rows.and.y%n_cols==p%n_cols,'multilabel metric target/prediction shapes differ')
    weights=metric_weights(y%n_rows,w);selected=selected_labels(y%n_cols,labels)
    if(samplewise) then
      allocate(tp(y%n_rows),true_count(y%n_rows),predicted(y%n_rows),multiplicity(y%n_cols));multiplicity=0
      do j=1,size(selected)
        multiplicity(selected(j)+1)=multiplicity(selected(j)+1)+1
      end do
    else
      allocate(tp(y%n_cols),true_count(y%n_cols),predicted(y%n_cols))
    end if
    tp=0;true_count=0;predicted=0
    do i=1,y%n_rows
      a=y%indptr(i);stop_a=y%indptr(i+1)-1;b=p%indptr(i);stop_b=p%indptr(i+1)-1
      ntrue=0;npred=0;ntp=0
      do j=a,stop_a
        field=y%indices(j)
        if(samplewise) then
          ntrue=ntrue+multiplicity(field)
        else
          true_count(field)=ieee_difference(true_count(field),-weights(i))
        end if
      end do
      do j=b,stop_b
        field=p%indices(j)
        if(samplewise) then
          npred=npred+multiplicity(field)
        else
          predicted(field)=ieee_difference(predicted(field),-weights(i))
        end if
      end do
      do while(a<=stop_a.and.b<=stop_b)
        if(y%indices(a)==p%indices(b)) then
          field=y%indices(a)
          if(samplewise) then
            ntp=ntp+multiplicity(field)
          else
            tp(field)=ieee_difference(tp(field),-weights(i))
          end if
          a=a+1;b=b+1
        else if(y%indices(a)<p%indices(b)) then
          a=a+1
        else
          b=b+1
        end if
      end do
      if(samplewise) then
        tp(i)=ieee_product(real(ntp,dp),weights(i));predicted(i)=ieee_product(real(npred,dp),weights(i))
        true_count(i)=ieee_product(real(ntrue,dp),weights(i))
      end if
    end do
    field=size(selected);if(samplewise) field=y%n_rows
    allocate(cm(field,2,2));total=numpy_sum(weights)
    do j=1,field
      i=j;if(.not.samplewise) i=selected(j)+1
      capacity=total;if(samplewise) capacity=ieee_product(real(size(selected),dp),weights(j))
      fp=ieee_difference(predicted(i),tp(i));fn=ieee_difference(true_count(i),tp(i))
      cm(j,1,1)=ieee_difference(ieee_difference(ieee_difference(capacity,tp(i)),fp),fn)
      cm(j,1,2)=fp;cm(j,2,1)=fn;cm(j,2,2)=tp(i)
    end do
  end function
  subroutine set_average(average,default,chosen)
    character(*), optional, intent(in) :: average
    character(*), intent(in) :: default
    character(:), allocatable, intent(out) :: chosen
    chosen=default;if(present(average)) chosen=trim(average)
    select case(chosen)
    case('none','binary','micro','macro','weighted','samples')
    case default;call require(.false.,'unknown classification metric average')
    end select
  end subroutine
  function metric_cm_vector(y,p,average,pos_label,labels,w) result(cm)
    integer, intent(in) :: y(:),p(:),pos_label
    character(*), intent(in) :: average
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: w(:)
    real(dp), allocatable :: cm(:,:,:)
    integer, allocatable :: present_labels(:)
    if(average=='binary') then
      present_labels=sorted_classes([y,p])
      call require(size(present_labels)<=2,'binary metric average requires binary targets')
      if(size(present_labels)==2) call require(any(present_labels==pos_label),'pos_label is not a valid observed label')
      cm=confusion_vector(y,p,[pos_label],w,.false.)
    else
      cm=confusion_vector(y,p,labels,w,average=='samples')
    end if
  end function
  function metric_cm_sparse(y,p,average,labels,w) result(cm)
    type(csr_matrix), intent(in) :: y,p
    character(*), intent(in) :: average
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: w(:)
    real(dp), allocatable :: cm(:,:,:)
    call require(average/='binary','binary metric average does not support multilabel targets')
    cm=confusion_sparse(y,p,labels,w,average=='samples')
  end function
  real(dp) function divide(numerator,denominator,zero) result(value)
    real(dp), intent(in) :: numerator,denominator,zero
    value=zero
    if(denominator/=0) value=ieee_quotient(numerator,denominator)
  end function
  real(dp) function reduce(values,weights,nan_average) result(value)
    real(dp), intent(in) :: values(:)
    real(dp), optional, intent(in) :: weights(:)
    logical, intent(in) :: nan_average
    real(dp), allocatable :: a(:),w(:)
    logical :: present_(size(values))
    if(nan_average) then
      present_=.not.ieee_is_nan(values);a=pack(values,present_)
      if(size(a)==0) then
        value=undefined();return
      end if
      if(present(weights)) then
        w=pack(weights,present_)
        if(numpy_sum(w)==0) then
          value=average_values(a)
        else
          value=average_values(a,w)
        end if
      else
        a=values
        where(.not.present_) a=0
        value=ieee_quotient(numpy_sum(a),real(count(present_),dp))
      end if
    else
      if(present(weights)) call require(numpy_sum(weights)/=0,'Jaccard averaging weights have zero total')
      value=average_values(values,weights)
    end if
  end function
  subroutine summarize(cm,average,beta,zero,w,pr,re,fs,ja,support)
    real(dp), intent(in) :: cm(:,:,:),beta,zero
    character(*), intent(in) :: average
    real(dp), optional, intent(in) :: w(:)
    real(dp), allocatable, intent(out) :: pr(:),re(:),fs(:),ja(:),support(:)
    real(dp), allocatable :: tp(:),predicted(:),truth(:),jd(:)
    real(dp) :: beta2,denom
    integer :: i
    call check_beta(beta)
    tp=cm(:,2,2);allocate(predicted(size(tp)),truth(size(tp)),jd(size(tp)))
    do i=1,size(tp)
      predicted(i)=ieee_difference(tp(i),-cm(i,1,2));truth(i)=ieee_difference(tp(i),-cm(i,2,1))
      jd(i)=ieee_difference(predicted(i),-cm(i,2,1))
    end do
    if(average=='micro') then
      tp=[numpy_sum(tp)];predicted=[numpy_sum(predicted)];truth=[numpy_sum(truth)];jd=[numpy_sum(jd)]
    end if
    allocate(pr(size(tp)),re(size(tp)),fs(size(tp)),ja(size(tp)));beta2=ieee_product(beta,beta)
    do i=1,size(tp)
      pr(i)=divide(tp(i),predicted(i),zero);re(i)=divide(tp(i),truth(i),zero)
      if(.not.ieee_is_finite(beta)) then
        fs(i)=re(i)
      else if(beta==0) then
        fs(i)=pr(i)
      else
        denom=ieee_difference(ieee_product(beta2,truth(i)),-predicted(i))
        fs(i)=divide(ieee_product(1+beta2,tp(i)),denom,zero)
      end if
      ja(i)=divide(tp(i),jd(i),zero)
    end do
    support=truth
  end subroutine
  subroutine averaged(pr,re,fs,ja,support,average,w)
    real(dp), allocatable, intent(inout) :: pr(:),re(:),fs(:),ja(:),support(:)
    character(*), intent(in) :: average
    real(dp), optional, intent(in) :: w(:)
    real(dp), allocatable :: weights(:)
    if(average=='none') return
    if(average=='weighted') then
      weights=support
    else if(average=='samples'.and.present(w)) then
      weights=w
    end if
    pr=[reduce(pr,weights,.true.)];re=[reduce(re,weights,.true.)];fs=[reduce(fs,weights,.true.)]
    ! Jaccard is reduced separately by the scalar score wrapper when requested.
    if(allocated(support)) deallocate(support)
  end subroutine
  real(dp) function precision_score_vv(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    integer, intent(in) :: y(:)
    integer, intent(in) :: p(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    cm=metric_cm_vector(y,p,avg,pos,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=pr
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(pr,weights_,.true.)
  end function
  real(dp) function recall_score_vv(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    integer, intent(in) :: y(:)
    integer, intent(in) :: p(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    cm=metric_cm_vector(y,p,avg,pos,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=re
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(re,weights_,.true.)
  end function
  real(dp) function f1_score_vv(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    integer, intent(in) :: y(:)
    integer, intent(in) :: p(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    cm=metric_cm_vector(y,p,avg,pos,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  real(dp) function jaccard_score_vv(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    integer, intent(in) :: y(:)
    integer, intent(in) :: p(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.true.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    cm=metric_cm_vector(y,p,avg,pos,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=ja
    if(avg=='weighted') then
      weights_=support_
      if(all(weights_==0)) deallocate(weights_)
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(ja,weights_,.false.)
  end function
  real(dp) function fbeta_score_vv(y,p,beta,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    integer, intent(in) :: y(:)
    integer, intent(in) :: p(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), intent(in) :: beta
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    b=beta
    call check_beta(b)
    cm=metric_cm_vector(y,p,avg,pos,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  subroutine precision_recall_fscore_support_vv(y,p,precision,recall,fscore,support,beta,labels,pos_label,average,sample_weight,zero_division)
    integer, intent(in) :: y(:)
    integer, intent(in) :: p(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), optional, intent(in) :: beta
    real(dp), allocatable, intent(out) :: precision(:),recall(:),fscore(:),support(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'none',avg)
    b=1
    if(present(beta)) b=beta
    call check_beta(b)
    cm=metric_cm_vector(y,p,avg,pos,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    call averaged(pr,re,fs,ja,support_,avg,sample_weight)
    precision=pr;recall=re;fscore=fs
    if(allocated(support_)) support=support_
  end subroutine
  function multilabel_confusion_matrix_vv(y,p,sample_weight,labels,samplewise) result(cm)
    integer, intent(in) :: y(:)
    integer, intent(in) :: p(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    logical, optional, intent(in) :: samplewise
    logical :: sw
    real(dp), allocatable :: cm(:,:,:)
    sw=.false.;if(present(samplewise)) sw=samplewise
    cm=confusion_vector(y,p,labels,sample_weight,sw)
  end function
  real(dp) function precision_score_bb(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    logical, intent(in) :: y(:)
    logical, intent(in) :: p(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    cm=metric_cm_vector(merge(1,0,y),merge(1,0,p),avg,pos,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=pr
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(pr,weights_,.true.)
  end function
  real(dp) function recall_score_bb(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    logical, intent(in) :: y(:)
    logical, intent(in) :: p(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    cm=metric_cm_vector(merge(1,0,y),merge(1,0,p),avg,pos,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=re
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(re,weights_,.true.)
  end function
  real(dp) function f1_score_bb(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    logical, intent(in) :: y(:)
    logical, intent(in) :: p(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    cm=metric_cm_vector(merge(1,0,y),merge(1,0,p),avg,pos,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  real(dp) function jaccard_score_bb(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    logical, intent(in) :: y(:)
    logical, intent(in) :: p(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.true.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    cm=metric_cm_vector(merge(1,0,y),merge(1,0,p),avg,pos,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=ja
    if(avg=='weighted') then
      weights_=support_
      if(all(weights_==0)) deallocate(weights_)
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(ja,weights_,.false.)
  end function
  real(dp) function fbeta_score_bb(y,p,beta,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    logical, intent(in) :: y(:)
    logical, intent(in) :: p(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), intent(in) :: beta
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    b=beta
    call check_beta(b)
    cm=metric_cm_vector(merge(1,0,y),merge(1,0,p),avg,pos,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  subroutine precision_recall_fscore_support_bb(y,p,precision,recall,fscore,support,beta,labels,pos_label,average,sample_weight,zero_division)
    logical, intent(in) :: y(:)
    logical, intent(in) :: p(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), optional, intent(in) :: beta
    real(dp), allocatable, intent(out) :: precision(:),recall(:),fscore(:),support(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'none',avg)
    b=1
    if(present(beta)) b=beta
    call check_beta(b)
    cm=metric_cm_vector(merge(1,0,y),merge(1,0,p),avg,pos,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    call averaged(pr,re,fs,ja,support_,avg,sample_weight)
    precision=pr;recall=re;fscore=fs
    if(allocated(support_)) support=support_
  end subroutine
  function multilabel_confusion_matrix_bb(y,p,sample_weight,labels,samplewise) result(cm)
    logical, intent(in) :: y(:)
    logical, intent(in) :: p(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    logical, optional, intent(in) :: samplewise
    logical :: sw
    real(dp), allocatable :: cm(:,:,:)
    sw=.false.;if(present(samplewise)) sw=samplewise
    cm=confusion_vector(merge(1,0,y),merge(1,0,p),labels,sample_weight,sw)
  end function
  real(dp) function precision_score_ii(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    integer, intent(in) :: y(:,:)
    integer, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    call require(all(shape(y)==shape(p)),'classification metric target/prediction shapes differ')
    call require(size(y,1)>0.and.size(y,2)>0,'classification metric matrices must be nonempty')
    if(size(y,2)==1) then
      cm=metric_cm_vector(y(:,1),p(:,1),avg,pos,labels,sample_weight)
    else
      sy=indicator_csr(y)
      sp=indicator_csr(p)
      cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    end if
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=pr
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(pr,weights_,.true.)
  end function
  real(dp) function recall_score_ii(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    integer, intent(in) :: y(:,:)
    integer, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    call require(all(shape(y)==shape(p)),'classification metric target/prediction shapes differ')
    call require(size(y,1)>0.and.size(y,2)>0,'classification metric matrices must be nonempty')
    if(size(y,2)==1) then
      cm=metric_cm_vector(y(:,1),p(:,1),avg,pos,labels,sample_weight)
    else
      sy=indicator_csr(y)
      sp=indicator_csr(p)
      cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    end if
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=re
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(re,weights_,.true.)
  end function
  real(dp) function f1_score_ii(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    integer, intent(in) :: y(:,:)
    integer, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    call require(all(shape(y)==shape(p)),'classification metric target/prediction shapes differ')
    call require(size(y,1)>0.and.size(y,2)>0,'classification metric matrices must be nonempty')
    if(size(y,2)==1) then
      cm=metric_cm_vector(y(:,1),p(:,1),avg,pos,labels,sample_weight)
    else
      sy=indicator_csr(y)
      sp=indicator_csr(p)
      cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    end if
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  real(dp) function jaccard_score_ii(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    integer, intent(in) :: y(:,:)
    integer, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.true.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    call require(all(shape(y)==shape(p)),'classification metric target/prediction shapes differ')
    call require(size(y,1)>0.and.size(y,2)>0,'classification metric matrices must be nonempty')
    if(size(y,2)==1) then
      cm=metric_cm_vector(y(:,1),p(:,1),avg,pos,labels,sample_weight)
    else
      sy=indicator_csr(y)
      sp=indicator_csr(p)
      cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    end if
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=ja
    if(avg=='weighted') then
      weights_=support_
      if(all(weights_==0)) deallocate(weights_)
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(ja,weights_,.false.)
  end function
  real(dp) function fbeta_score_ii(y,p,beta,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    integer, intent(in) :: y(:,:)
    integer, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), intent(in) :: beta
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    b=beta
    call check_beta(b)
    call require(all(shape(y)==shape(p)),'classification metric target/prediction shapes differ')
    call require(size(y,1)>0.and.size(y,2)>0,'classification metric matrices must be nonempty')
    if(size(y,2)==1) then
      cm=metric_cm_vector(y(:,1),p(:,1),avg,pos,labels,sample_weight)
    else
      sy=indicator_csr(y)
      sp=indicator_csr(p)
      cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    end if
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  subroutine precision_recall_fscore_support_ii(y,p,precision,recall,fscore,support,beta,labels,pos_label,average,sample_weight,zero_division)
    integer, intent(in) :: y(:,:)
    integer, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), optional, intent(in) :: beta
    real(dp), allocatable, intent(out) :: precision(:),recall(:),fscore(:),support(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'none',avg)
    b=1
    if(present(beta)) b=beta
    call check_beta(b)
    call require(all(shape(y)==shape(p)),'classification metric target/prediction shapes differ')
    call require(size(y,1)>0.and.size(y,2)>0,'classification metric matrices must be nonempty')
    if(size(y,2)==1) then
      cm=metric_cm_vector(y(:,1),p(:,1),avg,pos,labels,sample_weight)
    else
      sy=indicator_csr(y)
      sp=indicator_csr(p)
      cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    end if
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    call averaged(pr,re,fs,ja,support_,avg,sample_weight)
    precision=pr;recall=re;fscore=fs
    if(allocated(support_)) support=support_
  end subroutine
  function multilabel_confusion_matrix_ii(y,p,sample_weight,labels,samplewise) result(cm)
    integer, intent(in) :: y(:,:)
    integer, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    logical, optional, intent(in) :: samplewise
    logical :: sw
    real(dp), allocatable :: cm(:,:,:)
    type(csr_matrix) :: sy,sp
    sw=.false.;if(present(samplewise)) sw=samplewise
    call require(all(shape(y)==shape(p)),'classification metric target/prediction shapes differ')
    call require(size(y,1)>0.and.size(y,2)>0,'classification metric matrices must be nonempty')
    if(size(y,2)==1) then
      cm=confusion_vector(y(:,1),p(:,1),labels,sample_weight,sw)
    else
      sy=indicator_csr(y)
      sp=indicator_csr(p)
      cm=confusion_sparse(sy,sp,labels,sample_weight,sw)
    end if
  end function
  real(dp) function precision_score_ll(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    logical, intent(in) :: y(:,:)
    logical, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    call require(all(shape(y)==shape(p)),'classification metric target/prediction shapes differ')
    call require(size(y,1)>0.and.size(y,2)>0,'classification metric matrices must be nonempty')
    if(size(y,2)==1) then
      cm=metric_cm_vector(merge(1,0,y(:,1)),merge(1,0,p(:,1)),avg,pos,labels,sample_weight)
    else
      sy=indicator_csr(merge(1,0,y))
      sp=indicator_csr(merge(1,0,p))
      cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    end if
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=pr
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(pr,weights_,.true.)
  end function
  real(dp) function recall_score_ll(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    logical, intent(in) :: y(:,:)
    logical, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    call require(all(shape(y)==shape(p)),'classification metric target/prediction shapes differ')
    call require(size(y,1)>0.and.size(y,2)>0,'classification metric matrices must be nonempty')
    if(size(y,2)==1) then
      cm=metric_cm_vector(merge(1,0,y(:,1)),merge(1,0,p(:,1)),avg,pos,labels,sample_weight)
    else
      sy=indicator_csr(merge(1,0,y))
      sp=indicator_csr(merge(1,0,p))
      cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    end if
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=re
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(re,weights_,.true.)
  end function
  real(dp) function f1_score_ll(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    logical, intent(in) :: y(:,:)
    logical, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    call require(all(shape(y)==shape(p)),'classification metric target/prediction shapes differ')
    call require(size(y,1)>0.and.size(y,2)>0,'classification metric matrices must be nonempty')
    if(size(y,2)==1) then
      cm=metric_cm_vector(merge(1,0,y(:,1)),merge(1,0,p(:,1)),avg,pos,labels,sample_weight)
    else
      sy=indicator_csr(merge(1,0,y))
      sp=indicator_csr(merge(1,0,p))
      cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    end if
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  real(dp) function jaccard_score_ll(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    logical, intent(in) :: y(:,:)
    logical, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.true.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    call require(all(shape(y)==shape(p)),'classification metric target/prediction shapes differ')
    call require(size(y,1)>0.and.size(y,2)>0,'classification metric matrices must be nonempty')
    if(size(y,2)==1) then
      cm=metric_cm_vector(merge(1,0,y(:,1)),merge(1,0,p(:,1)),avg,pos,labels,sample_weight)
    else
      sy=indicator_csr(merge(1,0,y))
      sp=indicator_csr(merge(1,0,p))
      cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    end if
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=ja
    if(avg=='weighted') then
      weights_=support_
      if(all(weights_==0)) deallocate(weights_)
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(ja,weights_,.false.)
  end function
  real(dp) function fbeta_score_ll(y,p,beta,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    logical, intent(in) :: y(:,:)
    logical, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), intent(in) :: beta
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    b=beta
    call check_beta(b)
    call require(all(shape(y)==shape(p)),'classification metric target/prediction shapes differ')
    call require(size(y,1)>0.and.size(y,2)>0,'classification metric matrices must be nonempty')
    if(size(y,2)==1) then
      cm=metric_cm_vector(merge(1,0,y(:,1)),merge(1,0,p(:,1)),avg,pos,labels,sample_weight)
    else
      sy=indicator_csr(merge(1,0,y))
      sp=indicator_csr(merge(1,0,p))
      cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    end if
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  subroutine precision_recall_fscore_support_ll(y,p,precision,recall,fscore,support,beta,labels,pos_label,average,sample_weight,zero_division)
    logical, intent(in) :: y(:,:)
    logical, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), optional, intent(in) :: beta
    real(dp), allocatable, intent(out) :: precision(:),recall(:),fscore(:),support(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'none',avg)
    b=1
    if(present(beta)) b=beta
    call check_beta(b)
    call require(all(shape(y)==shape(p)),'classification metric target/prediction shapes differ')
    call require(size(y,1)>0.and.size(y,2)>0,'classification metric matrices must be nonempty')
    if(size(y,2)==1) then
      cm=metric_cm_vector(merge(1,0,y(:,1)),merge(1,0,p(:,1)),avg,pos,labels,sample_weight)
    else
      sy=indicator_csr(merge(1,0,y))
      sp=indicator_csr(merge(1,0,p))
      cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    end if
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    call averaged(pr,re,fs,ja,support_,avg,sample_weight)
    precision=pr;recall=re;fscore=fs
    if(allocated(support_)) support=support_
  end subroutine
  function multilabel_confusion_matrix_ll(y,p,sample_weight,labels,samplewise) result(cm)
    logical, intent(in) :: y(:,:)
    logical, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    logical, optional, intent(in) :: samplewise
    logical :: sw
    real(dp), allocatable :: cm(:,:,:)
    type(csr_matrix) :: sy,sp
    sw=.false.;if(present(samplewise)) sw=samplewise
    call require(all(shape(y)==shape(p)),'classification metric target/prediction shapes differ')
    call require(size(y,1)>0.and.size(y,2)>0,'classification metric matrices must be nonempty')
    if(size(y,2)==1) then
      cm=confusion_vector(merge(1,0,y(:,1)),merge(1,0,p(:,1)),labels,sample_weight,sw)
    else
      sy=indicator_csr(merge(1,0,y))
      sp=indicator_csr(merge(1,0,p))
      cm=confusion_sparse(sy,sp,labels,sample_weight,sw)
    end if
  end function
  real(dp) function precision_score_ss(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    type(csr_matrix), intent(in) :: y
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=y
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=pr
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(pr,weights_,.true.)
  end function
  real(dp) function recall_score_ss(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    type(csr_matrix), intent(in) :: y
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=y
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=re
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(re,weights_,.true.)
  end function
  real(dp) function f1_score_ss(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    type(csr_matrix), intent(in) :: y
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=y
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  real(dp) function jaccard_score_ss(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    type(csr_matrix), intent(in) :: y
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.true.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=y
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=ja
    if(avg=='weighted') then
      weights_=support_
      if(all(weights_==0)) deallocate(weights_)
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(ja,weights_,.false.)
  end function
  real(dp) function fbeta_score_ss(y,p,beta,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    type(csr_matrix), intent(in) :: y
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), intent(in) :: beta
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    b=beta
    call check_beta(b)
    sy=y
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  subroutine precision_recall_fscore_support_ss(y,p,precision,recall,fscore,support,beta,labels,pos_label,average,sample_weight,zero_division)
    type(csr_matrix), intent(in) :: y
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), optional, intent(in) :: beta
    real(dp), allocatable, intent(out) :: precision(:),recall(:),fscore(:),support(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'none',avg)
    b=1
    if(present(beta)) b=beta
    call check_beta(b)
    sy=y
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    call averaged(pr,re,fs,ja,support_,avg,sample_weight)
    precision=pr;recall=re;fscore=fs
    if(allocated(support_)) support=support_
  end subroutine
  function multilabel_confusion_matrix_ss(y,p,sample_weight,labels,samplewise) result(cm)
    type(csr_matrix), intent(in) :: y
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    logical, optional, intent(in) :: samplewise
    logical :: sw
    real(dp), allocatable :: cm(:,:,:)
    type(csr_matrix) :: sy,sp
    sw=.false.;if(present(samplewise)) sw=samplewise
    sy=y
    sp=p
    cm=confusion_sparse(sy,sp,labels,sample_weight,sw)
  end function
  real(dp) function precision_score_si(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    type(csr_matrix), intent(in) :: y
    integer, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=y
    sp=indicator_csr(p)
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=pr
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(pr,weights_,.true.)
  end function
  real(dp) function recall_score_si(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    type(csr_matrix), intent(in) :: y
    integer, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=y
    sp=indicator_csr(p)
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=re
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(re,weights_,.true.)
  end function
  real(dp) function f1_score_si(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    type(csr_matrix), intent(in) :: y
    integer, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=y
    sp=indicator_csr(p)
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  real(dp) function jaccard_score_si(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    type(csr_matrix), intent(in) :: y
    integer, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.true.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=y
    sp=indicator_csr(p)
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=ja
    if(avg=='weighted') then
      weights_=support_
      if(all(weights_==0)) deallocate(weights_)
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(ja,weights_,.false.)
  end function
  real(dp) function fbeta_score_si(y,p,beta,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    type(csr_matrix), intent(in) :: y
    integer, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), intent(in) :: beta
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    b=beta
    call check_beta(b)
    sy=y
    sp=indicator_csr(p)
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  subroutine precision_recall_fscore_support_si(y,p,precision,recall,fscore,support,beta,labels,pos_label,average,sample_weight,zero_division)
    type(csr_matrix), intent(in) :: y
    integer, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), optional, intent(in) :: beta
    real(dp), allocatable, intent(out) :: precision(:),recall(:),fscore(:),support(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'none',avg)
    b=1
    if(present(beta)) b=beta
    call check_beta(b)
    sy=y
    sp=indicator_csr(p)
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    call averaged(pr,re,fs,ja,support_,avg,sample_weight)
    precision=pr;recall=re;fscore=fs
    if(allocated(support_)) support=support_
  end subroutine
  function multilabel_confusion_matrix_si(y,p,sample_weight,labels,samplewise) result(cm)
    type(csr_matrix), intent(in) :: y
    integer, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    logical, optional, intent(in) :: samplewise
    logical :: sw
    real(dp), allocatable :: cm(:,:,:)
    type(csr_matrix) :: sy,sp
    sw=.false.;if(present(samplewise)) sw=samplewise
    sy=y
    sp=indicator_csr(p)
    cm=confusion_sparse(sy,sp,labels,sample_weight,sw)
  end function
  real(dp) function precision_score_is(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    integer, intent(in) :: y(:,:)
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=indicator_csr(y)
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=pr
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(pr,weights_,.true.)
  end function
  real(dp) function recall_score_is(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    integer, intent(in) :: y(:,:)
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=indicator_csr(y)
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=re
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(re,weights_,.true.)
  end function
  real(dp) function f1_score_is(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    integer, intent(in) :: y(:,:)
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=indicator_csr(y)
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  real(dp) function jaccard_score_is(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    integer, intent(in) :: y(:,:)
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.true.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=indicator_csr(y)
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=ja
    if(avg=='weighted') then
      weights_=support_
      if(all(weights_==0)) deallocate(weights_)
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(ja,weights_,.false.)
  end function
  real(dp) function fbeta_score_is(y,p,beta,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    integer, intent(in) :: y(:,:)
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), intent(in) :: beta
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    b=beta
    call check_beta(b)
    sy=indicator_csr(y)
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  subroutine precision_recall_fscore_support_is(y,p,precision,recall,fscore,support,beta,labels,pos_label,average,sample_weight,zero_division)
    integer, intent(in) :: y(:,:)
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), optional, intent(in) :: beta
    real(dp), allocatable, intent(out) :: precision(:),recall(:),fscore(:),support(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'none',avg)
    b=1
    if(present(beta)) b=beta
    call check_beta(b)
    sy=indicator_csr(y)
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    call averaged(pr,re,fs,ja,support_,avg,sample_weight)
    precision=pr;recall=re;fscore=fs
    if(allocated(support_)) support=support_
  end subroutine
  function multilabel_confusion_matrix_is(y,p,sample_weight,labels,samplewise) result(cm)
    integer, intent(in) :: y(:,:)
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    logical, optional, intent(in) :: samplewise
    logical :: sw
    real(dp), allocatable :: cm(:,:,:)
    type(csr_matrix) :: sy,sp
    sw=.false.;if(present(samplewise)) sw=samplewise
    sy=indicator_csr(y)
    sp=p
    cm=confusion_sparse(sy,sp,labels,sample_weight,sw)
  end function
  real(dp) function precision_score_sl(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    type(csr_matrix), intent(in) :: y
    logical, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=y
    sp=indicator_csr(merge(1,0,p))
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=pr
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(pr,weights_,.true.)
  end function
  real(dp) function recall_score_sl(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    type(csr_matrix), intent(in) :: y
    logical, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=y
    sp=indicator_csr(merge(1,0,p))
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=re
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(re,weights_,.true.)
  end function
  real(dp) function f1_score_sl(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    type(csr_matrix), intent(in) :: y
    logical, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=y
    sp=indicator_csr(merge(1,0,p))
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  real(dp) function jaccard_score_sl(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    type(csr_matrix), intent(in) :: y
    logical, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.true.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=y
    sp=indicator_csr(merge(1,0,p))
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=ja
    if(avg=='weighted') then
      weights_=support_
      if(all(weights_==0)) deallocate(weights_)
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(ja,weights_,.false.)
  end function
  real(dp) function fbeta_score_sl(y,p,beta,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    type(csr_matrix), intent(in) :: y
    logical, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), intent(in) :: beta
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    b=beta
    call check_beta(b)
    sy=y
    sp=indicator_csr(merge(1,0,p))
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  subroutine precision_recall_fscore_support_sl(y,p,precision,recall,fscore,support,beta,labels,pos_label,average,sample_weight,zero_division)
    type(csr_matrix), intent(in) :: y
    logical, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), optional, intent(in) :: beta
    real(dp), allocatable, intent(out) :: precision(:),recall(:),fscore(:),support(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'none',avg)
    b=1
    if(present(beta)) b=beta
    call check_beta(b)
    sy=y
    sp=indicator_csr(merge(1,0,p))
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    call averaged(pr,re,fs,ja,support_,avg,sample_weight)
    precision=pr;recall=re;fscore=fs
    if(allocated(support_)) support=support_
  end subroutine
  function multilabel_confusion_matrix_sl(y,p,sample_weight,labels,samplewise) result(cm)
    type(csr_matrix), intent(in) :: y
    logical, intent(in) :: p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    logical, optional, intent(in) :: samplewise
    logical :: sw
    real(dp), allocatable :: cm(:,:,:)
    type(csr_matrix) :: sy,sp
    sw=.false.;if(present(samplewise)) sw=samplewise
    sy=y
    sp=indicator_csr(merge(1,0,p))
    cm=confusion_sparse(sy,sp,labels,sample_weight,sw)
  end function
  real(dp) function precision_score_ls(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    logical, intent(in) :: y(:,:)
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=indicator_csr(merge(1,0,y))
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=pr
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(pr,weights_,.true.)
  end function
  real(dp) function recall_score_ls(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    logical, intent(in) :: y(:,:)
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=indicator_csr(merge(1,0,y))
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=re
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(re,weights_,.true.)
  end function
  real(dp) function f1_score_ls(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    logical, intent(in) :: y(:,:)
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=indicator_csr(merge(1,0,y))
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  real(dp) function jaccard_score_ls(y,p,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    logical, intent(in) :: y(:,:)
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.true.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    call check_beta(b)
    sy=indicator_csr(merge(1,0,y))
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=ja
    if(avg=='weighted') then
      weights_=support_
      if(all(weights_==0)) deallocate(weights_)
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(ja,weights_,.false.)
  end function
  real(dp) function fbeta_score_ls(y,p,beta,average,pos_label,sample_weight,labels,zero_division,raw_values) result(value)
    logical, intent(in) :: y(:,:)
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), intent(in) :: beta
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'binary',avg)
    b=1
    b=beta
    call check_beta(b)
    sy=indicator_csr(merge(1,0,y))
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    if(present(raw_values)) raw_values=fs
    if(avg=='weighted') then
      weights_=support_
    else if(avg=='samples'.and.present(sample_weight)) then
      weights_=sample_weight
    end if
    value=reduce(fs,weights_,.true.)
  end function
  subroutine precision_recall_fscore_support_ls(y,p,precision,recall,fscore,support,beta,labels,pos_label,average,sample_weight,zero_division)
    logical, intent(in) :: y(:,:)
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    character(*), optional, intent(in) :: average
    integer, optional, intent(in) :: pos_label
    real(dp), optional, intent(in) :: zero_division
    real(dp), optional, intent(in) :: beta
    real(dp), allocatable, intent(out) :: precision(:),recall(:),fscore(:),support(:)
    real(dp), allocatable :: cm(:,:,:),pr(:),re(:),fs(:),ja(:),support_(:),weights_(:)
    real(dp) :: b,z
    integer :: pos
    character(:), allocatable :: avg
    type(csr_matrix) :: sy,sp
    z=0;if(present(zero_division)) z=zero_division
    call check_zero(z,.false.)
    pos=1;if(present(pos_label)) pos=pos_label
    call set_average(average,'none',avg)
    b=1
    if(present(beta)) b=beta
    call check_beta(b)
    sy=indicator_csr(merge(1,0,y))
    sp=p
    cm=metric_cm_sparse(sy,sp,avg,labels,sample_weight)
    call summarize(cm,avg,b,z,sample_weight,pr,re,fs,ja,support_)
    call averaged(pr,re,fs,ja,support_,avg,sample_weight)
    precision=pr;recall=re;fscore=fs
    if(allocated(support_)) support=support_
  end subroutine
  function multilabel_confusion_matrix_ls(y,p,sample_weight,labels,samplewise) result(cm)
    logical, intent(in) :: y(:,:)
    type(csr_matrix), intent(in) :: p
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, optional, intent(in) :: labels(:)
    logical, optional, intent(in) :: samplewise
    logical :: sw
    real(dp), allocatable :: cm(:,:,:)
    type(csr_matrix) :: sy,sp
    sw=.false.;if(present(samplewise)) sw=samplewise
    sy=indicator_csr(merge(1,0,y))
    sp=p
    cm=confusion_sparse(sy,sp,labels,sample_weight,sw)
  end function
end module
