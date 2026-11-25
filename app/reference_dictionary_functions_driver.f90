program reference_dictionary_functions_driver
  use sklearn_common,only:dp
  use sklearn_dictionary,only:dict_learning,dict_learning_online,dictionary_learning_result, &
    online_dictionary_result,dictionary_callback,online_dictionary_callback
  use,intrinsic::iso_fortran_env,only:int64
  implicit none
  type(dictionary_learning_result)::offline
  type(online_dictionary_result)::online
  procedure(dictionary_callback),pointer::batch_callback
  procedure(online_dictionary_callback),pointer::stream_callback
  real(dp),allocatable::x(:,:),dictionary(:,:),code(:,:),events(:,:),errors(:),result_code(:,:),result_dict(:,:)
  real(dp)::alpha,tol
  integer::mode,n,p,k,limit,inner,seed,pd,pc,shuffle,batch,return_code,init,action,patience,status,i,nevents
  integer(int64)::iterations,steps
  character(4)::method
  do
    read(*,*,iostat=status)mode,n,p,k,limit,inner,seed,pd,pc,shuffle,batch,return_code,init,action,method,tol,patience,alpha
    if(status/=0)exit
    allocate(x(n,p),events(5,max(1,limit*((n-1)/max(1,batch)+1)+1)));nevents=0
    do i=1,n
      read(*,*)x(i,:)
    end do
    if(btest(init,0))then
      allocate(dictionary(max(1,k),p))
      do i=1,size(dictionary,1)
        read(*,*)dictionary(i,:)
      end do
    end if
    if(btest(init,1))then
      allocate(code(n,max(1,k)))
      do i=1,n
        read(*,*)code(i,:)
      end do
    end if
    nullify(batch_callback,stream_callback)
    if(action>=0)then
      batch_callback=>observe_batch;stream_callback=>observe_online
    end if
    if(mode==0)then
      offline=dict_learning(x,k,alpha,max_iter=limit,tol=tol,method=trim(method),dict_init=dictionary, &
        code_init=code,callback=batch_callback,random_state=seed,positive_dict=pd==1,positive_code=pc==1, &
        method_max_iter=inner)
      result_code=offline%code;result_dict=offline%dictionary;errors=offline%errors
      iterations=offline%n_iter;steps=0
    else
      online=dict_learning_online(x,k,alpha=alpha,max_iter=limit,return_code=return_code==1,dict_init=dictionary, &
        callback=stream_callback,batch_size=batch,shuffle=shuffle==1,method=trim(method),random_state=seed, &
        positive_dict=pd==1,positive_code=pc==1,method_max_iter=inner,tol=tol,max_no_improvement=patience)
      result_dict=online%dictionary;iterations=online%n_iter;steps=online%n_steps
      allocate(errors(0))
      if(allocated(online%code))then
        result_code=online%code
      else
        allocate(result_code(0,0))
      end if
    end if
    write(*,*)n,p,size(result_dict,1),iterations,steps,merge(1,0,size(result_code)>0),nevents,size(errors)
    write(*,'(*(es25.17e3,1x))')reshape(result_code,[size(result_code)])
    write(*,'(*(es25.17e3,1x))')reshape(result_dict,[size(result_dict)])
    write(*,'(*(es25.17e3,1x))')errors
    write(*,'(*(es25.17e3,1x))')reshape(events(:,:nevents),[5*nevents])
    deallocate(x,events,result_code,result_dict,errors)
    if(allocated(dictionary))deallocate(dictionary)
    if(allocated(code))deallocate(code)
    if(allocated(online%code))deallocate(online%code)
  end do
contains
  subroutine observe_batch(iteration,code,dictionary,errors)
    integer,intent(in)::iteration
    real(dp),intent(inout)::code(:,:),dictionary(:,:),errors(:)
    if(action==1.and.iteration==0)then
      dictionary=.99_dp*dictionary;errors(1)=1.001_dp*errors(1)
    end if
    nevents=nevents+1;events(:,nevents)=[real(iteration,dp),errors(size(errors)),sum(code),sum(dictionary),sum(errors)]
  end subroutine
  subroutine observe_online(step,cost,dictionary,a,b)
    integer(int64),intent(in)::step
    real(dp),intent(in)::cost
    real(dp),intent(inout)::dictionary(:,:),a(:,:),b(:,:)
    if(action==1.and.step==0)dictionary=.99_dp*dictionary
    nevents=nevents+1;events(:,nevents)=[real(step,dp),cost,sum(dictionary),sum(a),sum(b)]
  end subroutine
end program
