program reference_nb_sparse_driver
  use sklearn_common, only: dp
  use sklearn_sparse, only: csr_matrix,csr_from_arrays
  use sklearn_naive_bayes, only: multinomial_nb,bernoulli_nb,complement_nb
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  class(multinomial_nb), allocatable :: model
  type(csr_matrix) :: x,q,saved,piece
  real(dp), allocatable :: weights(:),dense(:,:)
  integer, allocatable :: y(:),classes(:),ends(:)
  integer :: kind,n,p,m,k,stream,batches,weighted,fitprior,fixed,norm,i,start,force,vector_alpha,dense_mask
  real(dp) :: alpha,threshold
  character(80) :: arg
  if(command_argument_count()>0) then
    call get_command_argument(1,arg);call reject(trim(arg));stop 2
  end if
  read(*,*) kind,n,p,m,k,stream,batches,weighted,fitprior,fixed,norm,alpha,threshold,force,vector_alpha,dense_mask
  select case(kind)
  case(0);allocate(multinomial_nb::model)
  case(1);allocate(bernoulli_nb::model)
  case(2);allocate(complement_nb::model)
  end select
  allocate(weights(n),y(n),classes(k),ends(batches))
  x=read_csr();q=read_csr();read(*,*) y;read(*,*) weights;read(*,*) classes;read(*,*) ends
  model%alpha=alpha;model%fit_prior=fitprior==1;model%force_alpha=force==1
  if(fixed==1) then
    allocate(model%class_prior(k));read(*,*) model%class_prior
  end if
  if(vector_alpha==1) then
    allocate(model%alpha_per_feature(p));read(*,*) model%alpha_per_feature
  end if
  select type(model)
  type is(bernoulli_nb);model%binarize_threshold=threshold
  type is(complement_nb);model%norm=norm==1
  end select
  saved=x
  if(stream==0) then
    if(btest(dense_mask,0)) then
      dense=x%to_dense()
      if(weighted==1) then
        call model%fit(dense,y,weights)
      else
        call model%fit(dense,y)
      end if
    else
      if(weighted==1) then
        call model%fit(x,y,weights)
      else
        call model%fit(x,y)
      end if
    end if
    call snapshot()
  else
    start=1
    do i=1,batches
      piece=subset(x,start,ends(i))
      if(btest(dense_mask,i-1)) then
        dense=piece%to_dense()
        if(i==1.or.mod(i,2)==0) then
          if(weighted==1) then
            call model%partial_fit(dense,y(start:ends(i)),classes,weights(start:ends(i)))
          else
            call model%partial_fit(dense,y(start:ends(i)),classes)
          end if
        else
          if(weighted==1) then
            call model%partial_fit(dense,y(start:ends(i)),sample_weight=weights(start:ends(i)))
          else
            call model%partial_fit(dense,y(start:ends(i)))
          end if
        end if
      else
        if(i==1.or.mod(i,2)==0) then
          if(weighted==1) then
            call model%partial_fit(piece,y(start:ends(i)),classes,weights(start:ends(i)))
          else
            call model%partial_fit(piece,y(start:ends(i)),classes)
          end if
        else
          if(weighted==1) then
            call model%partial_fit(piece,y(start:ends(i)),sample_weight=weights(start:ends(i)))
          else
            call model%partial_fit(piece,y(start:ends(i)))
          end if
        end if
      end if
      call snapshot();start=ends(i)+1
    end do
  end if
  if(any(x%data/=saved%data).or.any(x%indices/=saved%indices).or.any(x%indptr/=saved%indptr)) stop 4

contains
  function read_csr() result(matrix)
    type(csr_matrix) :: matrix
    integer :: count
    read(*,*) matrix%n_rows,matrix%n_cols,count
    allocate(matrix%indptr(matrix%n_rows+1),matrix%indices(count),matrix%data(count))
    read(*,*) matrix%indptr;read(*,*) matrix%indices;read(*,*) matrix%data
    call matrix%validate()
  end function
  function subset(matrix,first,last) result(result)
    type(csr_matrix), intent(in) :: matrix
    integer, intent(in) :: first,last
    type(csr_matrix) :: result
    integer :: begin,end
    begin=matrix%indptr(first);end=matrix%indptr(last+1)-1
    result=csr_from_arrays(last-first+1,matrix%n_cols,matrix%indptr(first:last+1)-begin+1, &
      matrix%indices(begin:end),matrix%data(begin:end))
  end function
  subroutine vector(a)
    real(dp), intent(in) :: a(:)
    print *,size(a);print '(*(es26.17e3,1x))',a
  end subroutine
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *,shape(a);print '(*(es26.17e3,1x))',reshape(a,[size(a)])
  end subroutine
  subroutine snapshot()
    call vector(real(model%classes_,dp));call matrix(model%feature_count_);call vector(model%class_count_)
    call matrix(model%feature_log_prob_);call vector(model%class_log_prior_)
    call matrix(model%joint_log_likelihood(q));call matrix(model%predict_log_proba(q));call matrix(model%predict_proba(q))
    call vector(real(model%predict(q),dp));call vector([real(model%n_features_in_,dp)]);call vector([model%score(x,y)]);call vector(real(model%predict(x),dp))
    dense=q%to_dense()
    call matrix(model%joint_log_likelihood(dense));call matrix(model%predict_log_proba(dense))
    call matrix(model%predict_proba(dense));call vector(real(model%predict(dense),dp))
    select type(model)
    type is(complement_nb);call vector(model%feature_all_)
    end select
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    type(multinomial_nb) :: mn
    type(bernoulli_nb) :: bn
    type(complement_nb) :: cn
    type(csr_matrix) :: matrix,query
    real(dp), allocatable :: probability(:,:)
    real(dp) :: bad
    integer :: labels(4)
    matrix=csr_from_arrays(4,3,[1,2,3,4,5],[1,2,1,3],[1._dp,2._dp,1._dp,2._dp])
    query=matrix;labels=[-2,-2,3,3];bad=ieee_value(0._dp,ieee_quiet_nan)
    select case(name)
    case('unallocated');deallocate(matrix%data)
    case('shape');matrix%n_rows=-1
    case('pointer_size');matrix%indptr=[1,2]
    case('pointer_order');matrix%indptr=[1,3,2,4,5]
    case('pointer_end');matrix%indptr(5)=4
    case('index_size');matrix%indices=[1,2]
    case('index_range');matrix%indices(1)=4
    case('empty_rows');matrix=csr_from_arrays(0,3,[1],[integer::],[real(dp)::])
    case('empty_columns');matrix=csr_from_arrays(4,0,[1,1,1,1,1],[integer::],[real(dp)::])
    case('nan_data');matrix%data(1)=bad
    case('inf_data');matrix%data(1)=ieee_value(0._dp,ieee_positive_inf)
    case('negative_data');matrix%data(1)=-1
    case('negative_complement');matrix%data(1)=-1;call cn%fit(matrix,labels);return
    case('bernoulli_nan');matrix%data(1)=bad;call bn%fit(matrix,labels);return
    case('first_classes');call mn%partial_fit(matrix,labels);return
    case('empty_classes');call mn%partial_fit(matrix,labels,[integer::]);return
    case('unknown_label');call mn%partial_fit(matrix,labels,[-2,4]);return
    case('changed_classes')
      call mn%fit(matrix,labels);call mn%partial_fit(matrix,labels,[-2,4]);return
    case('changed_width')
      call mn%fit(matrix,labels);query%n_cols=4;call mn%partial_fit(query,labels);return
    case('target_size');call mn%fit(matrix,labels(:3));return
    case('weight_size');call mn%fit(matrix,labels,[1._dp]);return
    case('negative_weight');call mn%fit(matrix,labels,[-1._dp,1._dp,1._dp,1._dp]);return
    case('nan_weight');call mn%fit(matrix,labels,[bad,1._dp,1._dp,1._dp]);return
    case('alpha_size');mn%alpha_per_feature=[1._dp,1._dp]
    case('alpha_negative');mn%alpha=-1
    case('threshold_negative');bn%binarize_threshold=-1;call bn%fit(matrix,labels);return
    case('unfitted');probability=mn%predict_proba(matrix);return
    case('query_width');call mn%fit(matrix,labels);query%n_cols=4;probability=mn%predict_proba(query);return
    case('query_nan');call mn%fit(matrix,labels);query%data(1)=bad;probability=mn%predict_proba(query);return
    case('query_negative');call mn%fit(matrix,labels);query%data(1)=-1;probability=mn%predict_proba(query);return
    case('product_range');matrix%data=2;call mn%fit(matrix,labels,spread(huge(1._dp),1,4));return
    case default;error stop 'unknown sparse NB rejection case'
    end select
    call mn%fit(matrix,labels)
  end subroutine
end program
